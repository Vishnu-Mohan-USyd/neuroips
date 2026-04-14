# R1+2 Debug Log

Persistent log of pure-R1+2 rescue attempts only. This excludes later non-R1+2 pivots such as shape-matched prediction.

Metric source for every row: `results/<run>/tuning_ring_recentered.json`, using `figures.recentered.stats` for `Relevant Expected` and `Relevant Unexpected`.

## Completed pure-R1+2 runs

| run | config | result dir | Rel Exp peak | Rel Unexp peak | Rel Exp FWHM | Rel Unexp FWHM | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---|
| `r12_base` | `config/sweep/sweep_rescue_1_2.yaml` | `results/r12_base` | 0.565376 | 0.639951 | 33.296673 | 31.745953 | Baseline R1+2 dampening: expected peak is lower, but the expected curve is still 1.550720° broader than unexpected. |
| `r12_w025` | `config/sweep/sweep_rescue_1_2_width_025.yaml` | `results/r12_w025` | 0.564079 | 0.638428 | 33.249197 | 31.746260 | `lambda_expected_width=0.25` is effectively base-like on the re-centered metric; no meaningful width rescue. |
| `r12_w050` | `config/sweep/sweep_rescue_1_2_width_050.yaml` | `results/r12_w050` | 0.561896 | 0.636785 | 33.279387 | 31.703062 | `lambda_expected_width=0.5` leaves the peak gap base-like and slightly worsens the re-centered width gap. |
| `r12_w100` | `config/sweep/sweep_rescue_1_2_width_100.yaml` | `results/r12_w100` | 0.563406 | 0.634037 | 33.211320 | 31.657583 | `lambda_expected_width=1.0` trims the peak gap slightly, but the expected curve remains 1.553737° broader than unexpected. |
| `r12_fb20` | `config/sweep/sweep_rescue_1_2_fb20.yaml` | `results/r12_fb20` | 0.583275 | 0.632221 | 32.158338 | 30.956241 | Adding fixed feedback surround materially shrinks both peak and width gaps versus base, but expected is still broader than unexpected. |
| `r12_fb24` | `config/sweep/sweep_rescue_1_2_fb24.yaml` | `results/r12_fb24` | 0.584996 | 0.633166 | 32.064951 | 30.858195 | Current best pure-R1+2 checkpoint per debugger: peak gap is the smallest so far and the width gap stays at the improved fb20 level. |

## Key findings carried forward

- Width-loss sweep finding: `r12_w025`, `r12_w050`, and `r12_w100` stay tightly clustered around the baseline re-centered gaps.
  Peak gap (`Rel Unexp peak - Rel Exp peak`) stays in the `0.070631` to `0.074889` range versus `0.074575` on `r12_base`.
  Width gap (`Rel Exp FWHM - Rel Unexp FWHM`) stays in the `1.502938°` to `1.576324°` range versus `1.550720°` on `r12_base`.
  This is consistent with the debugger conclusion that the expected-width loss is too small and/or misaligned with the re-centered target metric.
- Surround finding: `fb20` and `fb24` both improve the pure-R1+2 re-centered profile relative to base and the width-only sweep.
  `r12_fb24` reduces the relevant peak gap to `0.048170` and the relevant width gap to `1.206756°`, versus `0.074575` and `1.550720°` on `r12_base`.
- Current best pure-R1+2 checkpoint: keep `r12_fb24` as the comparison point for the next minimal loss-side variants.
  The recorded artifacts are consistent with that debugger-backed choice: `fb24` has the smallest relevant peak gap, while its width gap is effectively tied with `fb20`.

## Next pure-R1+2 candidates staged from `fb24`

- `config/sweep/sweep_rescue_1_2_fb24_localdisc_050.yaml`
  Minimal `fb24 + local_disc` candidate. Uses the branch's existing nonzero local-disc precedent (`lambda_local_disc: 0.5`) and the focused-only routing pattern already used in the routed dual sweeps.
- `config/sweep/sweep_rescue_1_2_fb24_sharp_050.yaml`
  Minimal `fb24 + sharp` candidate. Uses the branch's existing nonzero sharpness precedent (`lambda_sharp: 0.5`) and the same focused-only routing pattern.

## 2026-04-14 fb24 follow-up candidates

Both candidates completed training on CUDA with seed 42 and produced both analysis artifacts:
`results/<run>/tuning_ring_recentered.json` and `results/<run>/expected_vs_unexpected.json`.
Comparison point for this batch is `r12_fb24`.

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | Rel Exp FWHM | Rel Unexp FWHM | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `r12_fb24_localdisc_050` | `config/sweep/sweep_rescue_1_2_fb24_localdisc_050.yaml` | `results/r12_fb24_localdisc_050` | 4.110521 | 4.772126 | 0.596995 | 0.648338 | 32.373755 | 31.134261 | `local_disc=0.5` moves the pure-R1+2 checkpoint in the wrong direction on the re-centered metric: the peak gap widens to 0.051343 and the width gap widens to 1.239494°. |
| `r12_fb24_sharp_050` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050.yaml` | `results/r12_fb24_sharp_050` | 3.941245 | 4.547880 | 0.582977 | 0.626617 | 31.956252 | 30.794795 | `sharp=0.5` is the best pure-R1+2 result so far: the peak gap tightens to 0.043640 and the width gap tightens to 1.161456°, but expected is still broader than unexpected. |

### Batch take-away

- `r12_fb24_localdisc_050` is a regression against `r12_fb24` on the user’s target surface.
  Relative to `r12_fb24`, the relevant peak gap gets worse by `+0.003173` and the relevant width gap gets worse by `+0.032739°`.
- `r12_fb24_sharp_050` improves on `r12_fb24` on both relevant re-centered gaps.
  Relative to `r12_fb24`, the relevant peak gap improves by `-0.004531` and the relevant width gap improves by `-0.045299°`.
- Neither candidate meets the exact target `peak_exp < peak_unexp AND FWHM_exp < FWHM_unexp`.
  Both still satisfy the peak inequality and still fail the width inequality.
- Updated best pure-R1+2 checkpoint on the re-centered surface: `r12_fb24_sharp_050`.

## Debugger-backed next step after `r12_fb24_sharp_050`

- Accepted surface for this branch: relevant re-centered expected vs unexpected peak and FWHM from `tuning_ring_recentered.json`.
- Debugger conclusion carried forward:
  `sharp` is the only tested beneficial pure-R1+2 lever on that accepted surface.
  `local_disc` is ruled out on the accepted surface because `r12_fb24_localdisc_050` regressed both the relevant peak gap and the relevant width gap against `r12_fb24`.
- Current reference point: `r12_fb24_sharp_050`.
  It improves both relevant re-centered gaps versus `r12_fb24`, but expected FWHM is still larger than unexpected FWHM.
- Smallest next bracket around the current nonzero sharp setting:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_025.yaml`
  `config/sweep/sweep_rescue_1_2_fb24_sharp_075.yaml`
  These keep the `fb24` surround path and the focused-only sharp routing fixed, and only bracket `lambda_sharp` around `0.5`.

## Post-patch rationale: expected-width loss realigned to the accepted surface

- Debugger-backed rationale for the code change:
  the original `expected_width_loss` was a broad distance-weighted flank surrogate, so its gradient spread across the full outer ring instead of the shoulder that controls the accepted re-centered FWHM readout.
- Minimal R1+2-only replacement:
  keep the expected-only gating and the center dead zone, but penalize only shoulder-band activity above a per-trial half-max reference.
  With the current default `expected_width_deadzone_deg: 10.0`, the targeted shoulder is `10-20 deg`, which is the band that most directly moves the re-centered half-max crossing.
- First post-patch experiment should start from the current best pure-R1+2 base, `r12_fb24_sharp_050`, not from the old width-only branch.
  Recommended first config delta: clone `config/sweep/sweep_rescue_1_2_fb24_sharp_050.yaml` and set `lambda_expected_width: 0.25`.
  Reason: the aligned loss should be tested first at the smallest prior nonzero width weight, so we can detect shoulder-specific movement without immediately washing out the sharpness gains already present in `r12_fb24_sharp_050`.

## 2026-04-14 sharpness bracket around `r12_fb24_sharp_050`

Reference point for this bracket: `r12_fb24_sharp_050`.
Relevant re-centered gaps on the reference checkpoint are:
- peak gap (`Rel Unexp peak - Rel Exp peak`): `0.043640`
- width gap (`Rel Exp FWHM - Rel Unexp FWHM`): `1.161456°`

Both candidate runs completed training on CUDA with seed 42 and produced both analysis artifacts:
`results/<run>/tuning_ring_recentered.json` and `results/<run>/expected_vs_unexpected.json`.

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | Rel Exp FWHM | Rel Unexp FWHM | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `r12_fb24_sharp_025` | `config/sweep/sweep_rescue_1_2_fb24_sharp_025.yaml` | `results/r12_fb24_sharp_025` | 3.961381 | 4.579950 | 0.584019 | 0.629355 | 32.031683 | 30.874640 | `sharp=0.25` very slightly improves the width gap to 1.157044°, but peak gap regresses to 0.045336 versus the current best checkpoint. |
| `r12_fb24_sharp_075` | `config/sweep/sweep_rescue_1_2_fb24_sharp_075.yaml` | `results/r12_fb24_sharp_075` | 3.927192 | 4.533817 | 0.581338 | 0.625695 | 31.890861 | 30.711206 | `sharp=0.75` trims totals but worsens the width gap to 1.179655°; it does not beat the current best width gap. |

### Batch take-away

- `r12_fb24_sharp_025` beats the current best **width gap** by a narrow margin.
  Width gap improves from `1.161456°` on `r12_fb24_sharp_050` to `1.157044°` (`-0.004413°`).
  Peak gap gets worse from `0.043640` to `0.045336` (`+0.001696`).
- `r12_fb24_sharp_075` does **not** beat the current best width gap.
  Width gap worsens from `1.161456°` to `1.179655°` (`+0.018199°`).
- Neither candidate meets the exact target `peak_exp < peak_unexp AND FWHM_exp < FWHM_unexp`.
  Both still satisfy the peak inequality and still fail the width inequality.
- If optimizing **width gap only**, the updated best pure-R1+2 checkpoint is `r12_fb24_sharp_025`.
  If optimizing the joint peak+width tradeoff, `sharp_050` remains competitive because the width gain from `sharp_025` is extremely small while peak gap gets worse.

## Faraday magnitude-audit bracket for patched expected-width loss

- Base for this bracket: `config/sweep/sweep_rescue_1_2_fb24_sharp_050.yaml`.
  It is the cleanest current pure-R1+2 base because it already carries the best validated sharpness-side improvement without changing architecture.
- Faraday's magnitude-audit rationale:
  the earlier width-only sweep (`r12_w025`, `r12_w050`, `r12_w100`) was not informative about usable width magnitude because that loss body was misaligned with the accepted re-centered surface.
  After the patch, the loss is now shoulder-targeted and half-max-aligned, so the next step is not a broad sweep but a magnitude audit on the current best base.
- Bracket choice:
  `lambda_expected_width: 0.25` is the smallest prior nonzero width magnitude already used in this branch and is the low-risk probe for whether the aligned shoulder term moves the accepted width metric at all.
  `lambda_expected_width: 0.50` matches the current `lambda_sharp: 0.5`, giving a same-order comparison point without escalating to the previously over-broad `1.0` setting.
- New configs staged for this audit:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_025.yaml`
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_050.yaml`
  Both keep fb24 surround, focused sharp routing, and `lambda_sharp: 0.5` fixed; only `lambda_expected_width` changes.

## 2026-04-14 first post-patch `expected_width_loss` bracket

Patch context: shoulder-targeted `expected_width_loss` landed in the pure-R1+2 branch. These are the first two post-patch runs. Acceptance for this batch is judged only on the accepted re-centered relevant surface plus the debug-JSON controls:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_025` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_025.yaml` | `results/r12_fb24_sharp_050_width_025` | 3.906664 | 4.522081 | 0.579755 | 0.625847 | -0.046092 | 31.818874 | 30.642617 | 1.176257 | `NO` | `PASS` | `PASS` | Width-loss 0.25 preserves the negative peak gap but leaves the expected curve 1.176257° broader than unexpected; it fails the primary width criterion. |
| `r12_fb24_sharp_050_width_050` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_050.yaml` | `results/r12_fb24_sharp_050_width_050` | 3.868402 | 4.491299 | 0.577457 | 0.625062 | -0.047605 | 31.645051 | 30.585798 | 1.059253 | `NO` | `PASS` | `PASS` | Width-loss 0.50 is the best pure-R1+2 width result so far: peak gap is more negative and width gap tightens to 1.059253°, but expected remains broader than unexpected. |

### Batch take-away

- Neither post-patch candidate satisfies the full primary criterion because `width_gap` stays positive in both runs.
- Both post-patch candidates satisfy the secondary controls from the debug JSON.
  `L2/3 total exp < unexp` holds in both runs.
  `|L4 total diff| < 5%` also holds comfortably in both runs (`0.317%` and `0.453%`).
- `r12_fb24_sharp_050_width_050` is the new best pure-R1+2 checkpoint on the accepted re-centered width metric.
  Versus the prior width-gap leader `r12_fb24_sharp_025` (`1.157044°`), `width_050` improves the width gap to `1.059253°` (`-0.097790°`) and also makes the peak gap more negative (`-0.047605` vs `-0.045336`).

## Faraday next-step note after `r12_fb24_sharp_050_width_050`

- Evidence from the accepted surface:
  `r12_fb24_sharp_050_width_050` is the first stable post-patch improvement because it improves both primary relevant re-centered gaps against the prior best width-run.
  Relative to `r12_fb24_sharp_050_width_025`, peak gap becomes more negative (`-0.047605` vs `-0.046092`) and width gap tightens materially (`1.059253°` vs `1.176257°`), while the debug controls still pass.
- Why `0.75` is the smallest justified next step:
  `0.50` is now the best validated width magnitude on the accepted surface, but it still fails the primary width sign criterion because expected remains broader than unexpected.
  The next unseen increase above that successful setting is `0.75`; it is smaller than jumping straight back to the earlier `1.0` regime from the misaligned surrogate sweep, while still being large enough to test whether the aligned shoulder-targeted loss keeps improving from the new `0.50` result.
- New config staged from that evidence:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075.yaml`

## 2026-04-14 follow-up post-patch run `width_075`

Single-run follow-up on the shoulder-targeted `expected_width_loss` patch.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_075` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075.yaml` | `results/r12_fb24_sharp_050_width_075` | 3.830788 | 4.461732 | 0.574262 | 0.622488 | -0.048226 | 31.529939 | 30.544635 | 0.985304 | `NO` | `PASS` | `PASS` | Width-loss 0.75 further tightens the accepted width gap to 0.985304° and keeps the peak gap negative, but expected tuning is still broader than unexpected so the primary criterion still fails. |

### `width_075` take-away

- Primary criterion: `NO`.
  `peak_gap` is negative (`-0.048226`), but `width_gap` remains positive (`0.985304`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.855792 < 3.593729`).
  `|L4 total diff| < 5%` also holds (`0.166498%`).
- Relative to the prior best post-patch width run `r12_fb24_sharp_050_width_050`, `width_075` improves both primary scalar readouts on the accepted surface.
  `peak_gap` becomes more negative (`-0.048226` vs `-0.047605`).
  `width_gap` tightens from `1.059253°` to `0.985304°` (`-0.073949°`).
- Updated best pure-R1+2 checkpoint on the accepted width metric: `r12_fb24_sharp_050_width_075`.

## Faraday next-step note after `r12_fb24_sharp_050_width_075`

- Evidence from the accepted surface:
  `r12_fb24_sharp_050_width_075` still improves the accepted width gap and preserves the debug controls.
  Relative to `r12_fb24_sharp_050_width_050`, the width gap tightens from `1.059253°` to `0.985304°` (`-0.073949°`), the peak gap becomes more negative (`-0.048226` vs `-0.047605`), and both secondary controls remain `PASS`.
- Why `1.0` is the next smallest justified step:
  `0.75` is now the best validated width magnitude on the accepted surface, but it still fails the primary width sign criterion because expected remains broader than unexpected.
  The next unseen increase above that successful setting is `1.0`; it is the smallest remaining step in the branch's established width ladder and is justified by the continued monotonic improvement from `0.50` to `0.75`.
- New config staged from that evidence:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_100.yaml`

## 2026-04-14 follow-up post-patch run `width_100`

Single-run follow-up on the shoulder-targeted `expected_width_loss` patch at `lambda_expected_width=1.0`.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_100` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_100.yaml` | `results/r12_fb24_sharp_050_width_100` | 3.805884 | 4.439912 | 0.572609 | 0.621805 | -0.049195 | 31.443347 | 30.444021 | 0.999326 | `NO` | `PASS` | `PASS` | Width-loss 1.0 keeps the negative peak gap and passes both controls, but the expected curve is still broader than unexpected and the accepted width gap regresses slightly versus `width_075`. |

### `width_100` take-away

- Primary criterion: `NO`.
  `peak_gap` is negative (`-0.049195`), but `width_gap` remains positive (`0.999326`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.836823 < 3.567765`).
  `|L4 total diff| < 5%` also holds (`0.129544%`).
- Relative to the current best accepted width run `r12_fb24_sharp_050_width_075`, `width_100` makes the peak gap slightly more negative but slightly worsens the width gap.
  `peak_gap`: `-0.049195` vs `-0.048226`
  `width_gap`: `0.999326°` vs `0.985304°` (`+0.014022°` worse)
- Width conclusion after the `1.0` follow-up: within this tested family, `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on the accepted width metric.

## Faraday geometry note after `r12_fb24_sharp_050_width_075`

- Current best pure-R1+2 width checkpoint remains `r12_fb24_sharp_050_width_075`.
  Relevant expected FWHM is `31.529939°`, so the half-width is `15.764970°`.
- Geometry evidence from that accepted-surface statistic:
  the half-max crossing now sits just above `15°`, which means the `20°` bin is already below half-max and should not be the dominant remaining source of width error.
- Consequence for the patched shoulder-targeted loss:
  with `expected_width_deadzone_deg: 10.0`, the residual `10°` shoulder is still fully protected by the dead zone even though the measured half-max crossing is already outside `15°`.
  That makes the unpenalized `10°` bin the cleanest remaining geometric candidate for why expected tuning is still slightly too broad.
- Smallest justified next config step from that evidence:
  keep the current best magnitude (`lambda_expected_width: 0.75`) fixed and reduce `expected_width_deadzone_deg` from `10.0` to `5.0`.
  This leaves the rest of the pure-R1+2 recipe untouched while allowing the loss to act on the `10°` shoulder that is currently inside the protected center zone.
- New config staged from that evidence:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_deadzone_5.yaml`

## Faraday geometry rationale for explicit shoulder bounds

- Code-level change:
  the patched `expected_width_loss` now accepts explicit shoulder-band bounds, so the shoulder mask can be tested independently of `expected_width_deadzone_deg`.
  When those bounds are omitted, the legacy geometry is preserved exactly: lower = dead zone, upper = dead zone + 10°.
- Geometry evidence from the current best accepted-surface checkpoint `r12_fb24_sharp_050_width_075`:
  relevant expected FWHM is `31.529939°`, so the half-width is `15.764970°`.
  That places the current half-max crossing just above `15°`.
  The `20°` bin is therefore already below half-max, while the remaining unpenalized excess at `10°` still sits inside the current `10°` dead zone.
- First explicit geometry test:
  keep the current best base (`lambda_expected_width: 0.75`, `expected_width_deadzone_deg: 10.0`) fixed and isolate the `15°` shoulder only via an explicit `12.5-17.5°` band.
  This tests whether the accepted width gap is still being driven by the near-crossing `15°` shoulder, without yet changing the protected center or reintroducing pressure on the already-sub-half-max `20°` bin.
- New geometry-test config staged from that rationale:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_shoulder15.yaml`

## 2026-04-14 geometry-retune follow-up `width_075_deadzone_5`

Single-run geometry retune on the shoulder-targeted width setup: keep `lambda_expected_width=0.75` and reduce the expected-width deadzone to 5°.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_075_deadzone_5` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_deadzone_5.yaml` | `results/r12_fb24_sharp_050_width_075_deadzone_5` | 3.710898 | 4.339018 | 0.560950 | 0.612096 | -0.051147 | 31.310036 | 30.101303 | 1.208733 | `NO` | `PASS` | `PASS` | Deadzone 5° makes the peak gap more negative but materially worsens the accepted width gap versus `width_075`; expected remains broader than unexpected. |

### `width_075_deadzone_5` take-away

- Primary criterion: `NO`.
  `peak_gap` is negative (`-0.051147`), but `width_gap` remains positive (`1.208733`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.774048 < 3.480270`).
  `|L4 total diff| < 5%` also holds (`0.309938%`).
- Geometry-retune comparison to `r12_fb24_sharp_050_width_075`:
  `peak_gap` improves slightly (`-0.051147` vs `-0.048226`), but `width_gap` gets worse by `+0.223429°` (`1.208733°` vs `0.985304°`).
- Verdict on the geometry retune: it does **not** beat `width_075` on the accepted width metric.
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on that metric.

## 2026-04-14 geometry-retune follow-up `width_075_shoulder15`

Single-run geometry retune on the shoulder-targeted width setup: keep `lambda_expected_width=0.75` and isolate only the `15°` shoulder via an explicit `12.5-17.5°` band.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_075_shoulder15` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_shoulder15.yaml` | `results/r12_fb24_sharp_050_width_075_shoulder15` | 3.776789 | 4.412917 | 0.568008 | 0.622594 | -0.054586 | 31.414450 | 30.408112 | 1.006338 | `NO` | `PASS` | `PASS` | The explicit 15° shoulder band makes the peak gap more negative, but the accepted width gap remains positive and is slightly worse than `width_075`; expected stays broader than unexpected. |

### `width_075_shoulder15` take-away

- Primary criterion: `NO`.
  `peak_gap` is negative (`-0.054586`), but `width_gap` remains positive (`1.006338`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.821120 < 3.563862`).
  `|L4 total diff| < 5%` also holds (`0.093812%`).
- Geometry-retune comparison to `r12_fb24_sharp_050_width_075`:
  `peak_gap` improves (`-0.054586` vs `-0.048226`), but `width_gap` gets slightly worse by `+0.021034°` (`1.006338°` vs `0.985304°`).
- Verdict on the explicit shoulder-band retune: it does **not** beat `width_075` on the accepted width metric.
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on that metric.

## Faraday next-step note after the failed geometry retunes

- Current best width-loss settings to preserve:
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on the accepted width metric.
  The later geometry retunes both failed to beat it:
  `width_075_deadzone_5` worsened width gap to `1.208733°`,
  `width_075_shoulder15` worsened width gap to `1.006338°`,
  versus `0.985304°` on the unretuned `width_075` base.
- Debugger-backed rationale for the next lever:
  sharp is now the active broad-contraction lever, while `sigma_fb_surround` retuning is comparatively neutral on the accepted surface.
  Evidence:
  `fb20` vs `fb24` changed width gap only from `1.202097°` to `1.206756°` and peak gap only from `-0.048946` to `-0.048170`, which is effectively neutral at the scale of the later loss-side moves.
  By contrast, the sharp bracket on the shared `fb24` base moved width gap from `1.157044°` (`sharp_025`) to `1.179655°` (`sharp_075`), showing that sharp meaningfully changes the broad contraction profile once surround is fixed.
- Smallest next config step from that evidence:
  keep the current best width-loss settings fixed (`lambda_expected_width: 0.75`, implicit default shoulder geometry, no deadzone_5 or shoulder15 overrides) and lower `lambda_sharp` from `0.5` to `0.25`.
- New config staged from that rationale:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_025_width_075.yaml`

## 2026-04-14 follow-up `sharp_025_width_075`

Single-run follow-up on the current best width-loss checkpoint: keep `lambda_expected_width=0.75` fixed and relax the broad-contraction sharpness term from `0.5` to `0.25`.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_025_width_075` | `config/sweep/sweep_rescue_1_2_fb24_sharp_025_width_075.yaml` | `results/r12_fb24_sharp_025_width_075` | 3.853783 | 4.489767 | 0.576393 | 0.626049 | -0.049655 | 31.580705 | 30.554557 | 1.026148 | `NO` | `PASS` | `PASS` | Lowering `lambda_sharp` to 0.25 preserves the negative peak gap and both controls, but the accepted width gap gets worse versus `width_075`; expected remains broader than unexpected. |

### `sharp_025_width_075` take-away

- Primary criterion: `NO`.
  `peak_gap` is negative (`-0.049655`), but `width_gap` remains positive (`1.026148`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.863388 < 3.600976`).
  `|L4 total diff| < 5%` also holds (`0.278995%`).
- Comparison to `r12_fb24_sharp_050_width_075`:
  `peak_gap` improves slightly (`-0.049655` vs `-0.048226`), but `width_gap` worsens by `+0.040844°` (`1.026148°` vs `0.985304°`).
- Verdict on the sharpness relaxation: it does **not** beat `width_075` on the accepted width metric.
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on that metric.

## Faraday surround-bracket rationale after the sharp and geometry retunes

- Current best base to preserve:
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on the accepted width metric.
  The later sharp and geometry retunes all failed to beat it:
  `sharp_025_width_075` worsened width gap to `1.026148°`,
  `width_075_deadzone_5` worsened width gap to `1.208733°`,
  `width_075_shoulder15` worsened width gap to `1.006338°`,
  versus `0.985304°` on the unretuned `width_075` base.
- Debugger-backed rationale for the next remaining lever:
  surround is now the only remaining lever with strong completed evidence in this branch history.
  Completed surround runs already exist at `sigma_fb_surround=20` and `24`, while the more recent sharp and geometry retunes have directly shown how those loss-side changes behave and fail on the accepted surface.
- Why `20/22` is the smallest surround bracket supported by the run history:
  the completed surround history already spans `20` and `24`.
  On that completed bracket, the accepted-surface deltas were small:
  `fb20` width gap `1.202097°`, peak gap `-0.048946`
  `fb24` width gap `1.206756°`, peak gap `-0.048170`
  That makes `22` the smallest midpoint retune justified by existing completed runs, while retesting `20` on the current width_075 recipe gives the lower bracket edge on the same modern base.
- New configs staged from that rationale:
  `config/sweep/sweep_rescue_1_2_fb20_sharp_050_width_075.yaml`
  `config/sweep/sweep_rescue_1_2_fb22_sharp_050_width_075.yaml`

## Faraday recurrence-spread rationale after the surround bracket

- Current best base to preserve:
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on the accepted width metric.
  The surround bracket did not beat it:
  `fb20_sharp_050_width_075` worsened width gap to `1.135758°`,
  `fb22_sharp_050_width_075` worsened width gap to `1.108323°`,
  versus `0.985304°` on the `fb24_sharp_050_width_075` base.
- Debugger-backed recurrence rationale:
  the residual width problem is not explained by a large feedforward mismatch, because L4 stays nearly matched while the L2/3 control asymmetry persists even with feedback removed.
  On the current best base, the debug JSON reports `l4_control_percent_diff = 0.166498%`, while the `feedback_off_control` block still shows `expected_l23_mean = 1.178180` and `unexpected_l23_mean = 1.062162` (`FB-OFF gap = -0.116019` for unexp-exp).
  That makes local L2/3 recurrence spread the first recurrence-side lever to test before touching gain or feedback again.
- Smallest recurrence-spread step from that evidence:
  keep `gain_rec = 0.3` and all feedback-side settings fixed, and reduce `sigma_rec` from `15.0` to `11.0`.
- New config staged from that rationale:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec11.yaml`

## 2026-04-14 surround bracket follow-up `fb20_width_075` and `fb22_width_075`

Two-run surround bracket around the current width-loss base: keep `lambda_sharp=0.5` and `lambda_expected_width=0.75` fixed while lowering `sigma_fb_surround` from `24.0` to `20.0` and `22.0`.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb20_sharp_050_width_075` | `config/sweep/sweep_rescue_1_2_fb20_sharp_050_width_075.yaml` | `results/r12_fb20_sharp_050_width_075` | 3.834919 | 4.464758 | 0.571212 | 0.623419 | -0.052207 | 31.675124 | 30.539367 | 1.135758 | `NO` | `PASS` | `PASS` | Lowering `sigma_fb_surround` to 20.0 makes the peak gap more negative but materially worsens the accepted width gap versus `width_075`; expected remains broader than unexpected. |
| `r12_fb22_sharp_050_width_075` | `config/sweep/sweep_rescue_1_2_fb22_sharp_050_width_075.yaml` | `results/r12_fb22_sharp_050_width_075` | 3.832342 | 4.463127 | 0.572460 | 0.622690 | -0.050230 | 31.632569 | 30.524246 | 1.108323 | `NO` | `PASS` | `PASS` | Lowering `sigma_fb_surround` to 22.0 also keeps the negative peak gap and both controls, but the accepted width gap is still worse than `width_075`. |

### surround bracket take-away

- Primary criterion: `NO` for both runs.
  `fb20`: `peak_gap=-0.052207`, `width_gap=1.135758`.
  `fb22`: `peak_gap=-0.050230`, `width_gap=1.108323`.
- Secondary controls: `PASS` for both runs.
  `fb20`: `L2/3 total 2.829970 < 3.598855`, `|L4 diff|=0.351008%`.
  `fb22`: `L2/3 total 2.849104 < 3.593306`, `|L4 diff|=0.022445%`.
- Comparison to `r12_fb24_sharp_050_width_075`:
  reference `peak_gap=-0.048226`, `width_gap=0.985304`.
  `fb20` improves peak gap but worsens width gap by `+0.150453°`.
  `fb22` improves peak gap but worsens width gap by `+0.123019°`.
- Verdict on the surround bracket: neither `fb20` nor `fb22` beats `width_075` on the accepted width metric.
  `r12_fb24_sharp_050_width_075` remains the best pure-R1+2 checkpoint on that metric.

## 2026-04-14 recurrence-width follow-up `fb24_sharp_050_width_075_rec11`

Single-run recurrence retune on the current width-loss base: keep `sigma_fb_surround=24.0`, `lambda_sharp=0.5`, and `lambda_expected_width=0.75` fixed while lowering `sigma_rec` to `11.0`.
Acceptance remains:
- primary: `peak_gap = exp_peak - unexp_peak < 0`
- primary: `width_gap = exp_fwhm - unexp_fwhm < 0`
- secondary control: `L2/3 total exp < unexp`
- secondary control: `|L4 total diff| < 5%`

| run | config | result dir | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | primary pass? | L2/3 total control | L4 total control | one-line interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `r12_fb24_sharp_050_width_075_rec11` | `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec11.yaml` | `results/r12_fb24_sharp_050_width_075_rec11` | 3.763561 | 4.360018 | 0.584107 | 0.601886 | -0.017779 | 30.413189 | 30.244571 | 0.168618 | `NO` | `PASS` | `PASS` | Lowering `sigma_rec` to 11.0 sharply improves the accepted width gap while preserving predictive suppression and the L4 control, but expected is still slightly broader than unexpected on the accepted surface. |

### `fb24_sharp_050_width_075_rec11` take-away

- Primary criterion: `NO`.
  `peak_gap` remains negative (`-0.017779`), but `width_gap` is still positive (`0.168618`).
- Secondary controls: both `PASS`.
  `L2/3 total exp < unexp` holds (`2.851151 < 3.534590`).
  `|L4 total diff| < 5%` also holds (`1.489150%`).
- Comparison to `r12_fb24_sharp_050_width_075`:
  reference `peak_gap=-0.048226`, `width_gap=0.985304`.
  `rec11` makes the peak gap less negative by `+0.030446`, but improves the width gap by `-0.816686°` (`0.168618°` vs `0.985304°`).
- Verdict on the recurrence retune: it does **not** satisfy the full primary target yet, but it **does** beat `width_075` as the current best pure-R1+2 checkpoint on the accepted width metric while keeping the secondary controls intact.

## Faraday recurrence bracket after `rec11`

- Confirmed completed-evidence basis:
  `r12_fb24_sharp_050_width_075_rec11` is the only local retune so far that materially improved the accepted `width_gap`, moving it from `0.985304°` on `r12_fb24_sharp_050_width_075` down to `0.168618°` while keeping both secondary controls intact.
- Confirmed tradeoff that motivates the bracket:
  the same `rec11` run weakened peak suppression, with `peak_gap` moving from `-0.048226` to `-0.017779`.
  That makes recurrence spread the only completed local retune with strong positive width evidence, but it also shows a live risk that further contraction could collapse the negative peak gap.
- Smallest next recurrence bracket from that evidence:
  keep `gain_rec = 0.3` and all feedback-side settings fixed, and lower only `model.sigma_rec` from `11.0` to `10.5` and `10.0`.
  The purpose of this bracket is to test whether the accepted `width_gap` flips negative before peak suppression collapses.
- New configs staged from that rationale:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec105.yaml`
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec10.yaml`

## Faraday training-side alignment after the delta acceptance fix

- Confirmed mismatch between training and acceptance surfaces:
  the corrected acceptance analysis now scores evoked response
  `Δr = r_l23[t_readout] - r_l23[t_isi_last]`
  on an eval-style focused expected bucket derived from prediction error at the
  previous ISI. Before this patch, the expected-linked training losses still
  used raw late-ON windows and generator mismatch labels.
- Minimal training-side alignment that keeps unrelated objectives unchanged:
  leave mismatch, sharp, local discrimination, prediction suppression, and all
  other losses on their current paths.
  Change only `expected_suppress` and `expected_width` so they use:
  1. an aligned expected mask from previous-ISI `q_pred`
  2. non-ambiguous presentations only
  3. focused/relevant task-state only
  4. evoked response `late_ON - prev_ISI` instead of raw late-ON state
- First retrain target from the current best recurrence base:
  `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec11_aligned.yaml`
  This is a name-only derivative of `rec11`; the retrain isolates the
  training-surface alignment change without changing any hyperparameters.

## Faraday analysis-side correction: raw-state contamination

- Confirmed debugger diagnosis:
  the current acceptance surface in `scripts/plot_tuning_ring_extended.py` scores the raw late-ON L2/3 state, so the width/peak readout is contaminated by whatever residual state already exists at the end of the preceding ISI.
- Smallest accepted-surface correction:
  keep the same trial buckets, same re-centering, and same FWHM logic, but add an explicit delta-scored mode that measures
  `Δr = r_l23[t_readout] - r_l23[t_isi_last]`
  on the same trials.
- Why this is the first analysis-side fix:
  it does not change training, checkpoints, or bucket definitions.
  It only makes the acceptance readout auditable by separating evoked response from inherited baseline state.

## 2026-04-14 delta-mode corrected-surface audit `width_075` and `rec11`

Targeted validation for the delta-response analysis patch in `scripts/plot_tuning_ring_extended.py`.
Test command:
- `pytest -q tests/test_plot_tuning_ring_extended.py`
Result:
- `2 passed in 1.16s`

Corrected-surface rerun uses `--response-mode delta`, i.e. `r_l23[t_readout] - r_l23[t_isi_last]`, on the incumbent checkpoints.
Acceptance on this corrected surface:
- `peak_gap = exp_peak - unexp_peak < 0`
- `width_gap = exp_fwhm - unexp_fwhm < 0`

| run | checkpoint | delta JSON | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | corrected-surface pass? |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `r12_fb24_sharp_050_width_075` | `results/r12_fb24_sharp_050_width_075/emergent_seed42/checkpoint.pt` | `results/r12_fb24_sharp_050_width_075/tuning_ring_recentered_delta.json` | 0.144178 | 0.323438 | -0.179260 | 27.524199 | 28.988576 | -1.464377 | `YES` |
| `r12_fb24_sharp_050_width_075_rec11` | `results/r12_fb24_sharp_050_width_075_rec11/emergent_seed42/checkpoint.pt` | `results/r12_fb24_sharp_050_width_075_rec11/tuning_ring_recentered_delta.json` | 0.149112 | 0.323028 | -0.173916 | 26.144043 | 29.012496 | -2.868453 | `YES` |

### delta-mode take-away

- On the corrected delta-response surface, **both** incumbent checkpoints satisfy the strict geometry target:
  `width_gap < 0` and `peak_gap < 0`.
- `rec11` remains the stronger checkpoint on the corrected width metric.
  Compared with the base `width_075`, it widens the negative width gap from `-1.464377°` to `-2.868453°`.

## 2026-04-14 aligned retrain follow-up `rec11_aligned`

Targeted validation for the training-side alignment patch.
Test command:
- `pytest -q tests/test_training.py -k "AlignedExpectedTrainingPath or ExpectedSuppressFeatureSpecific or ExpectedWidthLoss"`
Result:
- `8 passed, 79 deselected in 1.07s`

Aligned retrain:
- config: `config/sweep/sweep_rescue_1_2_fb24_sharp_050_width_075_rec11_aligned.yaml`
- output: `results/r12_fb24_sharp_050_width_075_rec11_aligned`
- seed/device: `42`, `cuda`

### incumbent `rec11` vs aligned retrain

| surface | run | Rel Exp total | Rel Unexp total | Rel Exp peak | Rel Unexp peak | peak_gap | Rel Exp FWHM | Rel Unexp FWHM | width_gap | peak_gap<0? | width_gap<0? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| raw | incumbent `rec11` | 3.763561 | 4.360018 | 0.584107 | 0.601886 | -0.017779 | 30.413189 | 30.244571 | 0.168618 | `YES` | `NO` |
| raw | aligned `rec11` | 3.816031 | 5.022787 | 0.575212 | 0.591233 | -0.016021 | 31.222711 | 32.595634 | -1.372923 | `YES` | `YES` |
| delta | incumbent `rec11` | 0.505282 | 1.276897 | 0.148518 | 0.322521 | -0.174003 | 26.197003 | 29.021329 | -2.824326 | `YES` | `YES` |
| delta | aligned `rec11` | 0.182007 | 1.743990 | 0.097282 | 0.431171 | -0.333889 | 25.005761 | 32.848499 | -7.842738 | `YES` | `YES` |

### controls

| run | L2/3 exp | L2/3 unexp | L2/3 exp<unexp? | L4 exp | L4 unexp | |L4 diff| % | L4 control pass? |
|---|---:|---:|---|---:|---:|---:|---|
| incumbent `rec11` | 2.851151 | 3.534590 | `YES` | 1.812177 | 1.839163 | 1.489150 | `YES` |
| aligned `rec11` | 3.641309 | 4.877433 | `YES` | 1.755696 | 1.784938 | 1.665516 | `YES` |

### aligned take-away

- On the corrected delta surface, the aligned retrain preserves the qualitative result and strengthens it.
  `peak_gap` stays negative and becomes more negative (`-0.333889` vs `-0.174003`).
  `width_gap` stays negative and becomes much more negative (`-7.842738°` vs `-2.824326°`).
- On the old raw surface, the aligned retrain changes the qualitative result in the favorable direction.
  The incumbent `rec11` raw surface had `width_gap=+0.168618°`.
  The aligned retrain raw surface flips that to `width_gap=-1.372923°` while keeping `peak_gap=-0.016021 < 0`.
- Peak-gap sign check:
  `peak_gap < 0` holds on both surfaces for the aligned retrain.

### analysis note: explicit baseline surface

- `scripts/plot_tuning_ring_extended.py` now also supports
  `--response-mode baseline`, defined as `r_l23[t_isi_last]` on the same
  buckets, re-centering, and FWHM logic as the raw and delta surfaces.
- Purpose:
  compare the pre-stimulus inherited state directly against the evoked
  `delta` surface without changing any training or bucket semantics.

### analysis note: matched-history ring comparison

- `scripts/plot_tuning_ring_extended.py` now also supports
  `--match-history preprobe_observed`.
- Rationale:
  expected and unexpected ring averages can differ simply because the
  observable pre-probe history is distributed differently across the two
  classes.
  The matched-history mode keeps the current expected/unexpected definition
  unchanged, but only compares trials from contexts present in both classes
  within a regime, and weights trials so each shared context contributes
  equally within each class.

### analysis note: branch-point counterfactual probes

- `scripts/plot_tuning_ring_extended.py` now also supports
  `--comparison-mode branch_counterfactual`.
- Rationale:
  matched-history still compares different realized trials.
  The branch-point mode freezes the exact full recurrent state at the last ISI
  timestep, then branches from that identical state into two probe stimuli.
  That removes pre-probe-state mismatch entirely.
- Implementation choice:
  expected probe = the `q_pred` peak at the frozen branch point.
  unexpected probe = expected probe + `90°`.
  This reuses the repo’s existing hidden-state probe precedent for a
  maximally unexpected counterfactual branch.

### fix note: branch-counterfactual baseline centering

- Bug:
  in `--comparison-mode branch_counterfactual --response-mode baseline`,
  both branches correctly returned the same frozen pre-probe `r_l23`, but the
  collector still stored different `true_ch` values for expected vs unexpected.
  The later re-centering step rolled those identical baselines by different
  amounts, which made the same baseline state appear artificially opposite.
- Minimal fix:
  baseline mode now uses the shared expected/predicted channel as the
  centering key for both counterfactual branches.
  Raw and delta branch modes keep their original expected/unexpected probe
  centering unchanged.
