# Hardening Pass Results

After external code review identified 6 bugs and 4 confound risks, a systematic
hardening pass was performed on the sharpening-investigation branch. This document
summarizes what was fixed, what was verified, and what survives.

## Phase A: Bug fixes

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `lambda_sharp` not parsed from YAML | P1 ran with lambda_sharp=0 (silent no-op). Previous P1 result invalid. | Added to `load_config()` in src/config.py |
| 2 | Oracle used `metadata.states` (current) instead of `true_states` (next) | q_pred wrong on ~5% of state-switch trials | Use true_states in stage2_feedback.py |
| 3 | Readout window `max(1, steps_on-3)` gave only 2 steps when steps_on=3 | Sharpening experiments read barely-settled L2/3 | Changed to `max(0, steps_on-3)` |
| 4 | Orientation decoder trained during Stage 2 mechanistic runs | "Sensory accuracy" confounded moving-probe effect with representation | Added `freeze_decoder` config; auto-freeze when `freeze_v2=True` |
| 5 | State BCE loss active in oracle mode | Meaningless gradient from p_cw=0.5 placeholder against true states | Pass None to loss in oracle mode; set lambda_state=0 in all freeze_v2 configs |
| 6 | Stage 1 gating not mandatory | Could silently proceed to Stage 2 on bad V1 circuit | Added `--allow-gating-fail` flag; RuntimeError if gating fails by default |

All 326 tests pass (322 existing + 4 new regression tests for these bugs).

## Phase B: Verification reruns (1 seed, sharpening-investigation fixes applied)

Compared pre-fix vs post-fix for the three headline conditions:

| Condition | Pre-fix ‖alpha‖ | Post-fix ‖alpha‖ | Pre SI(0°) pi=5 | Post SI(0°) pi=5 | Verdict |
|---|---|---|---|---|---|
| Deviance (dampening) | 2.147 | 2.147 | +22.5% | +38.0% | Unchanged (SI higher because decoder is now frozen; gradient no longer absorbs effect) |
| P1 (tuning sharpness loss) | 0.195 | **1.603** | +0.7% at 45° | **+6.7% at 45°, +9.0% at 90°** | **Previously broken (lambda_sharp=0)**. Now produces broad surround suppression. |
| P4 (fine discrim, steps_on=3→6) | 1.213 | 1.511 | +4.1% at 30° | +4.2% at 30°, **+4.8% at 45°** | Modestly stronger with more L2/3 settling time. |

## Phase C: Confound controls (1 seed each)

Tests whether the learned effects depend on:
- **Adaptation** (`alpha_adaptation=0`): is the result an L4 adaptation artifact?
- **Transition reliability** (`p_transition_cw=p_transition_ccw=0.5`): does it require informative predictions? At 50% reliability, CW and CCW are equally likely so the oracle prediction is at chance and the feedback template is effectively uncorrelated with the upcoming stimulus.

### Dampening controls

| Condition | ‖alpha‖ | R(dampening) | SI(0°) pi=5 | SI(10°) pi=5 |
|---|---|---|---|---|
| Original | 2.147 | +0.914 | +38.0% | +25.6% |
| No adaptation | 2.148 | +0.914 | +37.7% | +25.4% |
| 50%-reliability transitions | 2.178 | +0.914 | +38.6% | +26.0% |

**Dampening is unaffected by either control.** Within numerical noise across all three conditions.
Conclusion: dampening in this **minimal V1-V2 inhibitory feedback model with laminar populations** is **not** driven by expectation. It is a generic energy-minimization effect that exploits whatever prediction template (real or random) is supplied, to suppress L2/3 activity at the q_pred peak where the SOM drive is most concentrated.

### Sharpening (P4) controls

| Condition | ‖alpha‖ | R(sharpening) | Peak SI offset | Peak SI pi=5 |
|---|---|---|---|---|
| Original | 1.511 | +0.968 | 45° | +4.8% |
| No adaptation | 1.569 | +0.970 | 45° | +4.8% |
| 50%-reliability transitions | 1.617 | +0.773 | 30° | +5.4% |

**Sharpening survives no-adaptation (identical to original).**
**Under 50%-reliability transitions, the kernel shifts:** R(sharpening) drops 0.97 -> 0.77, the peak moves from 45° to 30°, and the suppression at 45° drops from +4.8% to +1.7%. The profile is still sharpening-like but the shape changes. Part of the P4 sharpening depends on the prediction structure; part is driven purely by fine discrimination + noise pressure.

## Phase D: End-to-end learned V2 (1 seed each)

Tests whether the effects survive when V2 learns its own predictions instead of receiving oracle ground truth.

| Condition | Mode | V2 cw_acc | ‖alpha‖ | R(damp) | R(sharp) | SI(0°) pi=5 | SI(45°) pi=5 |
|---|---|---|---|---|---|---|---|
| Dampening | oracle | - | 2.147 | +0.914 | -0.902 | +38.0% | +0.0% |
| Dampening | e2e | 0.438 | **2.336** | +0.915 | -0.903 | **+42.4%** | +0.0% |
| P4 sharpening | oracle | - | 1.511 | -0.813 | +0.968 | +0.0% | +4.8% |
| P4 sharpening | e2e | 0.441 | 0.807 | **+0.924** | **-0.917** | **+15.0%** | +0.0% |

**Dampening survives end-to-end.** Even with a near-chance V2 (cw_acc=43.8%), dampening emerges slightly stronger. This is consistent with the confound findings — dampening doesn't need good predictions.

**Sharpening does NOT survive end-to-end.** With learned V2 (cw_acc=44.1%), the P4 profile completely flips from sharpening to **dampening** (R(damp)=+0.92, SI(0°)=+15%). The sharpening profile depends on having precise predictions; when V2 is weak, the system defaults to the energy-minimization regime (dampening).

## Key findings

1. **Dampening is robust but not a prediction-error signal.** It survives:
   - Fixing the state alignment bug
   - Freezing the decoder
   - Removing adaptation
   - 50%-reliability transitions
   - Learned V2 (even at near-chance)
   
   But it is **not** expectation-driven. It's an energy-minimization effect that uses any prediction template to concentrate SOM suppression on the most active channels.

2. **Sharpening is fragile.** It emerges only under specific conditions (fine discrimination + noise + oracle V2). When any of these break down — particularly when V2 is learned and imperfect — the system falls back to dampening.

3. **The "objective determines regime" claim needs narrowing.** The original framing was "sensory-loss off gives dampening; sensory-loss on gives sharpening." The hardened picture is: "sensory-loss off gives dampening (generic energy); sensory-loss on + fine discrimination + oracle V2 gives sharpening (fragile); with learned V2, everything tends toward dampening."

4. **What the model actually supports**: A mechanistic/normative claim that under a specific set of conditions (fine discrimination, high noise, oracle predictions), an inhibitory feedback operator can learn a surround-suppression kernel that implements sharpening in L2/3. This is a narrower but defensible result.

## Results location

- `results/hardening/` — Phase B reruns
- `results/confounds/` — Phase C confound controls
- `results/e2e/` — Phase D end-to-end runs

Note on result directory names: earlier runs were saved under `results/confounds/damp_no_predict_s42/` and `results/confounds/p4_no_predict_s42/`. Those directories are kept as-is; the corresponding configs have been renamed from `confound_{damp,p4}_no_predict.yaml` to `confound_{damp,p4}_50reliable.yaml` to more accurately describe the manipulation (50%-reliability transitions, not "unpredictable").

## Configs

- `config/exp_deviance.yaml` — dampening (sensory off, mismatch on L2/3)
- `config/exp_sharp_p1.yaml` — sharpening via `lambda_sharp`
- `config/exp_sharp_p3.yaml` — high pi + no energy + sensory only
- `config/exp_sharp_p4.yaml` — fine discrimination + noise
- `config/confound_damp_no_adapt.yaml` — dampening with `alpha_adaptation=0`
- `config/confound_damp_50reliable.yaml` — dampening with 50%-reliability transitions
- `config/confound_p4_no_adapt.yaml` — P4 with `alpha_adaptation=0`
- `config/confound_p4_50reliable.yaml` — P4 with 50%-reliability transitions
- `config/e2e_deviance.yaml` — dampening with learned V2
- `config/e2e_p4.yaml` — P4 with learned V2
