# Rate/Homeostasis Remediation Report

Date: 2026-06-08  
Branch: `v1-rate-homeostasis-remediation`  
Checkpoint before remediation: `0ce46a4`  
Accepted candidate: `v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix`

## Scope

This report documents the local rate/homeostasis remediation work around natural-video L4 drive, L2/3 reliability, PV/SOM stabilization, and strict matrix validation. It is not a claim of full biological perfection. The accepted state is a constrained engineering/biology compromise: it improves L2/3 reliability and preserves the existing V1 validation suite while keeping explicit no-cheat guards, but L4E video/event rates remain high and PV causality is only clean under PV-off ablation, not under the stricter half-PV criterion.

## Code Changes Made

All code changes were made in `genn/v1TwoLayerModel.cc`; validators/tests were not edited.

1. Video and analytic L4 drive scale knobs.
   - Added `V1_VIDEO_L4_DRIVE_SCALE`, default `1.0`, applied only to video/frame L4E current paths.
   - Added `V1_ANALYTIC_L4_DRIVE_SCALE`, default `1.0`, applied only to analytic/grating/context assay L4E drive paths.
   - Summary/no-cheat rows record that these scale knobs do not use future frames, labels, heldout frames, or output assemblies.
   - Accepted candidate uses `V1_VIDEO_L4_DRIVE_SCALE=0.85` and `V1_ANALYTIC_L4_DRIVE_SCALE=1.0`.

2. Video L4 divisive normalization attempts.
   - Added opt-in video-only local temporal divisive normalization with `V1_VIDEO_L4_DIVISIVE_NORM_ENABLE`, `BETA`, `SIGMA`, `TAU_MS`, and `RADIUS`.
   - Initial implementation divided the full source current; Debugger evidence showed raw KITTI L4 drive had a floor/min/median around `0.12 nA`, so whole-current division erased the operating point and could collapse reliability.
   - Patched to contrast-only/floor-preserving normalization with `V1_VIDEO_L4_DIVISIVE_NORM_FLOOR_NA=0.12`: pool and normalize only `max(source - floor, 0)`, then output `floor + contrast / denominator`.
   - Patched again to use one-frame-lagged normalization state: current frame denominator comes from previous `norm_state`, and only after writing the frame is `norm_state` updated toward current local contrast energy.
   - Accepted candidate keeps divisive normalization disabled: `video_l4_divisive_norm_enabled=0.000000`.

3. L4E spike-frequency adaptation.
   - Added opt-in L4E-local adaptation using existing `V1LIF` `AdaptCurrent`, `TauAdapt`, and `AdaptSpike`.
   - Knobs: `V1_L4E_ADAPTATION_ENABLE`, `V1_L4E_ADAPTATION_TAU_MS`, `V1_L4E_ADAPTATION_SPIKE_NA`.
   - This is per-cell local spike-frequency adaptation inside the GeNN neuron model, not host-side global scaling.
   - Accepted candidate keeps it disabled: `l4e_adaptation_enabled=0.000000`; defaults logged as `tau_ms=250.000000`, `spike_na=0.000500`.

4. Final accepted PV/SOM reliability and homeostatic settings.
   - `video_pv_reliability_tuning_enabled=1.000000`
   - `video_pv_reliability_output_scale=1.050000`
   - `video_pv_reliability_l23pv_to_l23e_only=1.000000`
   - `video_som_reliability_tuning_enabled=1.000000`
   - `video_som_reliability_output_scale=0.900000`
   - `video_som_reliability_l23som_to_l23e_only=1.000000`
   - `post_video_inhibitory_stabilization_pv_eta_scale=1.100000`
   - `post_video_inhibitory_stabilization_pv_target_hz=22.500000`
   - `post_video_inhibitory_stabilization_tail_gate_post_cell_fraction=0.325586`
   - `post_video_inhibitory_stabilization_l23pv_to_l23e_changed_frac=0.329602`
   - `post_video_inhibitory_stabilization_l23pv_to_l23e_p95_abs_delta=0.000972`
   - `post_video_inhibitory_stabilization_l23som_to_l23e_changed_frac=0.276305`
   - `post_video_inhibitory_stabilization_l23som_to_l23e_p95_abs_delta=0.001045`

## Accepted Candidate Configuration

The accepted run is `v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090`.

Key configuration rows from the accepted summary:

| Metric | Value |
| --- | ---: |
| `validation_sheet_side` | `40` |
| `validation_core_enabled` | `1.000000` |
| `validation_core_side` | `32` |
| `video_frame_count` | `64` |
| `video_repeat_count` | `3` |
| `video_presentation_count` | `192` |
| `video_frame_ms` | `100.000000` |
| `video_l4_drive_scale` | `0.850000` |
| `analytic_l4_drive_scale` | `1.000000` |
| `video_l4_divisive_norm_enabled` | `0.000000` |
| `l4e_adaptation_enabled` | `0.000000` |
| `l4_l23_orientation_bias_strength` | `0.000000` |
| `l4_l23_feedforward_orientation_prior_enabled` | `0.000000` |
| `l4_l23_orientation_neutral_density_match_active` | `1.000000` |
| `video_ff_stdp_aplus` | `0.000120` |
| `video_ff_stdp_aminus` | `0.000105` |
| `video_ff_homeostatic_scaling_scale` | `1.500000` |
| `video_ff_bcm_competition_enabled` | `1.000000` |
| `video_ff_event_trace_enabled` | `1.000000` |
| `video_recurrent_only_consolidation_pass_count` | `3` |
| `video_recurrent_only_consolidation_l23ee_stdp_aplus` | `0.000100` |
| `video_recurrent_only_consolidation_l23ee_stdp_aminus` | `0.000115` |

## Key Validation Metrics

### Natural-video L2/3 reliability

Validator source: `v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_strict_matrix_validator_pvoff_for_pvcausality.log`.

| Validation | Result |
| --- | --- |
| `l23_video_representational_validity` | PASS: same-similarity `0.727344`, different-similarity `0.177638`, gap `0.549706`; frame top-1 `1.000000`, top-5 `1.000000`, chance `0.015625` |
| `l23_video_l4_l23_geometry_alignment` | PASS: RSM correlation `0.559760`, temporal shuffle `0.336504`, spatial shuffle `0.229678` |
| `l23_video_raw_topk_repeat_oracle_ceiling` | PASS: leave-one-repeat-out top-5 oracle recall `0.622917`; leaky repeat mean `0.741667` |
| `l23_activity_raw_topk_repeat_stability` | PASS: raw oracle@5 `0.622917`, threshold `0.600000`; L2/3 repeat correlation `0.717396`; raw oracle ceiling fraction `0.839888` |
| `raw_oracle_0p6_milestone` | PASS: raw oracle@5 `0.622917`, top-k `5`, repeat count `3`, sample count `192` |
| `l23_activity_anti_cheat_separation` | PASS: raw exact gate passed; not rescued by frame decoding or smoothed population metrics |

Natural-video frame-rate summary from `*_video_frame_summary.csv`:

| Population / quantity | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| L4E video rate Hz | `81.402887` | `78.091406` | `85.355078` |
| L2/3E video rate Hz | `0.515712` | `0.418359` | `0.648828` |
| L2/3PV video rate Hz | `2.419238` | `1.393750` | `3.693750` |
| L2/3SOM video rate Hz | `16.730111` | `15.306250` | `18.318750` |
| L4 drive min/mean/max/std | min mean `0.120000`, mean `0.194298`, max mean `1.000000`, std mean `0.141856` |

Natural-video/event timing note from `*_video_event_population_bins.csv`:

| Metric | Value |
| --- | ---: |
| L4E event count | `384` |
| L4E event peak-rate mean | `125.926565 Hz` |
| L4E event peak-time mean | `18.041667 ms` |
| L4E first post-stimulus nonzero bin mean | `0.000000 ms` |

### Orientation, coverage, and no-cheat gates

| Validation | Result |
| --- | --- |
| Core L2/3 OSI | PASS: final-post-video L2/3 core OSI `0.841802`; control `0.000000`; delta `0.841802`; pre-video full post `0.358986` |
| Sheet summary OSI | baseline L4 `0.270700`; baseline L2/3 `0.000000`; post L4 `0.271020`; post L2/3 `0.666667`; final-post-video L2/3 `0.772935` |
| L2/3 sparse responsiveness | PASS: peak >=5 Hz fraction `0.224121`; peak >=10 Hz fraction `0.174866`; total core cells `16384` |
| L2/3 multiphase responsiveness | PASS: peak-any-phase >=5 Hz fraction `0.338806`; responsive median phase-pooled OSI `0.916515` |
| L2/3 responsive site coverage | PASS: multiphase sites >=1 fraction `0.936523`; >=2 fraction `0.855469`; single-phase site fraction `0.861328` |
| L2/3 sparse population rates | PASS: fraction below 1 Hz `0.893555`; p99 `1.614583 Hz`; limit `5.000000 Hz` |
| Spatial coverage balance | PASS: responsive sites `959`; responsive cells `5551`; min quadrant site fraction `0.242961`; min quadrant cell fraction `0.243199` |
| No-hardcode audit | PASS: `l4_l23_orientation_bias_strength=0.000000`, `feedforward_orientation_prior_enabled=0.000000`, `inhibitory_orientation_rule_enabled=0.000000` |

### SOM/context and size tuning

| Validation | Result |
| --- | --- |
| SOM broad/full suppression | PASS: final-post-video mean BSI `0.678187` |
| SOM-off causal effect | PASS: full mean BSI `0.678187`, SOM-off mean BSI `0.286742`, delta `0.391445` |
| SOM size tuning | PASS: peak radius `2.000000`; L2/3E rates `[0.000000, 46.331019, 67.129630, 13.784722, 0.254630, 0.011574]` for radii `[0.5,1,2,3,4,6]` |
| Large-field suppression | PASS: peak rate `67.129630`, large rate `0.011574`, suppression `0.999828` |
| SOM-off size rescue | PASS: full suppression `0.999828`, SOM-off suppression `0.672750`, delta `0.327077`; SOM-off large L2/3E rate `30.092593 Hz` |
| Orientation-context SOM-off effect | PASS: full mean OSD `0.702175`, SOM-off mean OSD `0.197428`, OSD reduction `0.504747`; same-suppression reduction `0.693895` |

### PV gain and inhibitory stability

There are two relevant PV validations:

1. Strict half-PV criterion, using `pvweak_scale=0.500000`, failed:
   - `full_mean_l23e_hz=27.654272`
   - `pvweak_mean_l23e_hz=31.293513`
   - `mean_increase_fraction=0.131598`
   - required `0.200000`
   - This means the model did not meet the strict "half-PV should increase L2/3E by at least 20%" criterion.

2. PV-off causality, using `pvweak_scale=0.000000`, passed:
   - `full_mean_l23e_hz=27.654272`
   - `pvweak_mean_l23e_hz=32.424842`
   - `mean_increase_fraction=0.172508`
   - required `0.100000`
   - `full_median_l23e_hz=27.187500`
   - `pvweak_median_l23e_hz=32.187500`
   - `median_increase_fraction=0.183908`
   - selectivity safety passed: full median OSI `0.423265`, PV-off median OSI `0.424168`, median OSI drop `-0.000903`
   - PV rates passed: full L2/3PV post median `61.250000 Hz`, p99 `101.250000 Hz`, limit `150.000000 Hz`

Interpretation: PV participates in gain control, and removing it causally increases L2/3E responses without degrading selectivity. The half-PV criterion is stricter and was not met by this candidate; therefore the accepted claim should be "PV-off causality passes; half-PV criterion remains unresolved," not "PV gain is fully biologically matched."

### Recurrent L2/3 validation

| Validation | Result |
| --- | --- |
| L23E->L23E plasticity | PASS: active `643887`; changed fraction `0.087526`; p95 abs delta `0.000433`; bounds maintained |
| Response-correlation specificity | PASS: high-correlation endpoints had larger delta and final weights than low-correlation endpoints; high mean delta `0.000093`, low mean delta `0.000022`; high mean weight `0.004591`, low mean weight `0.004489` |
| Strong-synapse enrichment | PASS: top recurrent weights corr>0.2 fraction `0.459121`, all fraction `0.410449`; odds ratio `1.246886`; combined coactive/co-tuned odds ratio `1.301617` |
| Heavy-tail-like bounded distribution | PASS: p50 `0.004214`, p95 `0.007784`, p99 `0.009915`, max `0.010000`, gini `0.207043`, top10 mass share `0.179273` |
| Shuffle specificity | PASS: observed top fraction delta `0.048672`; shuffle q95 delta `0.012613`; z-score `7.013226` |
| Recurrence contribution | PASS: recurrence-on mean corr `0.201665`, recurrence-off mean corr `0.184807`, delta `0.016857`; corr>0.2 fraction on `0.327032`, off `0.297995`, delta `0.029037` |
| Rate/OSI safety | PASS: mean peak on `9.018149`, off `8.628101`; peak ratio off/on `0.956749`; mean OSI on `0.558984`, off `0.515698` |

## Evidence-Driven Rejected Mechanisms

1. Raw whole-current video L4 divisive normalization.
   - Initial norm divided the entire source current, including the `0.12 nA` KITTI drive floor.
   - Evidence: high beta run `v1_ratehomeo_videol4scale085_divnorm080_full_5090` collapsed natural-video activity: L2/3E, L2/3PV, and L2/3SOM frame rates went to zero, raw oracle was `nan`, and natural-video delay/reliability gates failed.
   - Conclusion: normalizing the floor erased the operating point; this mechanism is implemented only as an opt-in corrected version and was not selected for the accepted candidate.

2. Weak whole-current divnorm.
   - `divnorm003` and `divnorm008` preserved some OSI but failed raw oracle 0.6 and matrix causality controls.
   - `divnorm003`: raw oracle@5 `0.590625`, below `0.600000`; SOM/PV/recurrence ablation matrix failures remained.
   - `divnorm008`: raw oracle@5 `0.578125`, below `0.600000`; similar causality/control failures.
   - Conclusion: weak whole-current normalization did not solve reliability and still interfered with matrix validation.

3. Corrected floor-preserving/lagged divnorm.
   - Implemented as an opt-in mechanism after Debugger evidence, but not accepted yet.
   - Summary evidence:
     - `divnorm_floor080_lag`: final-post-video L2/3 median OSI `0.766885`.
     - `divnorm_floor020_lag`: final-post-video L2/3 median OSI `0.768164`.
   - These were not selected as the accepted matrix candidate because the accepted candidate already passed the strict reliability target with simpler video scaling and PV/SOM homeostasis, while divnorm needed further matrix validation before claiming benefit.

4. L4E adaptation at `AdaptSpike=0.0005 nA`.
   - Implemented as opt-in per-cell adaptation using `V1LIF`, not a host-side hack.
   - Run `v1_ratehomeo_videol4scale085_l4eadapt0005_full_5090` had final-post-video L2/3 median OSI `0.759062`, lower than the accepted matrix full run's sheet summary `0.772935`.
   - It was not selected because it did not clearly improve the matrix target and would add another active mechanism without enough evidence.

5. Analytic L4 scale reduction.
   - Earlier analytic scale sweeps could pass raw oracle in some cases, but disturbed analytic validation balance and/or failed matrix controls.
   - Example `analyticscale090_rerun`: raw oracle@5 `0.508333`, active tile fraction too high (`0.766113` mean, `0.875000` max), OSI gate failed with control higher than full (`full_post=0.815195`, `control_post=0.846834`, delta `-0.031639`).
   - Accepted candidate keeps analytic scale at `1.0` to avoid tuning analytic/grating assays to rescue video reliability.

6. L4E->L23PV or L4PV-only scaling alone.
   - Debugger evidence before the accepted matrix showed L4E rates were invariant/high across PV-scale tests and L4E->L23PV scaling alone did not produce a clean matrix pass.
   - This was replaced by the accepted combination of video L4 scale `0.85`, learned feedforward gain/homeostatic scaling, L2/3 recurrent consolidation, and PV/SOM reliability/homeostatic settings.

## No-Cheat Statement

The accepted candidate does not use top-down/HVA feedback, output assemblies, validation targets, labels, heldout frames, or future-frame targets to form the L2/3 representation.

Summary/validator evidence:

- `l4_l23_orientation_bias_strength=0.000000`
- `l4_l23_feedforward_orientation_prior_enabled=0.000000`
- `inhibitory_orientation_rule_enabled=0.000000`
- `video_l4_drive_scale_future_frame_used=0.000000`
- `video_l4_drive_scale_target_label_used=0.000000`
- `video_l4_drive_scale_heldout_frames_used=0.000000`
- `video_l4_drive_scale_output_assembly_used=0.000000`
- `video_ff_stdp_future_frame_used=0.000000`
- `video_ff_stdp_target_label_used=0.000000`
- `video_ff_stdp_heldout_frames_used=0.000000`
- `video_ff_homeostatic_scaling_future_frame_used=0.000000`
- `video_ff_homeostatic_scaling_target_label_used=0.000000`
- `video_ff_homeostatic_scaling_heldout_frames_used=0.000000`
- `video_ff_bcm_competition_future_frame_used=0.000000`
- `video_ff_bcm_competition_target_label_used=0.000000`
- `video_ff_bcm_competition_orientation_label_used=0.000000`
- `video_ff_bcm_competition_heldout_frames_used=0.000000`
- `video_ff_event_trace_future_frame_used=0.000000`
- `video_ff_event_trace_target_label_used=0.000000`
- `video_ff_event_trace_heldout_frames_used=0.000000`
- `post_video_inhibitory_stabilization_future_frame_used=0.000000`
- `post_video_inhibitory_stabilization_target_label_used=0.000000`
- `post_video_inhibitory_stabilization_orientation_label_used=0.000000`
- `post_video_inhibitory_stabilization_heldout_frames_used=0.000000`
- `post_video_inhibitory_stabilization_output_assembly_used=0.000000`
- `l23_activity_anti_cheat_separation`: PASS, raw exact gate passed and was not rescued by frame decoding or smoothed population metrics.

## Remaining Caveats

1. L4E video/event firing remains high.
   - Natural-video L4E frame mean was `81.402887 Hz`, with frame range `78.091406-85.355078 Hz`.
   - Event L4E peak-rate mean was `125.926565 Hz`.
   - Analytic size-tuning validation also reports high L4 peak rate `156.863426 Hz`.
   - This is a clear remaining mismatch relative to conservative biological firing-rate expectations, even though downstream L2/3 sparse-rate gates pass.

2. Half-PV gain criterion remains unresolved.
   - Half-PV run failed the strict criterion: mean L2/3E increase `0.131598` versus required `0.200000`.
   - PV-off causality passed: mean increase `0.172508` versus required `0.100000`, with selectivity safety preserved.
   - Report PV as causally involved, not fully quantitatively matched.

3. Corrected divisive normalization is implemented but not the accepted candidate.
   - The final accepted matrix run keeps `video_l4_divisive_norm_enabled=0`.
   - The corrected contrast-only, floor-preserving, one-frame-lagged divnorm remains available for future testing, but should not be described as validated by the accepted candidate.

4. L4E adaptation is implemented but not active in the accepted candidate.
   - It remains an opt-in biological mechanism but was not selected by evidence in this remediation pass.

5. The model is still a reduced two-layer V1 scaffold.
   - VIP/top-down expectation circuitry is not active here.
   - No claim is made that all macaque/cat V1 firing rates, laminar physiology, dendritic targeting, conductance dynamics, or full developmental biology are reproduced.

## Log and Artifact Paths

Accepted run artifacts:

- Full run log: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090.log`
- Full run directory: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/`
- Full summary CSV: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_summary.csv`
- Video frame summary: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_video_frame_summary.csv`
- Video site rates: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_video_site_rates.csv`
- Video event bins: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_video_event_population_bins.csv`
- Video consolidation metrics: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_video_consolidation_metrics.csv`
- SOM context validation: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_som_context_validation.csv`
- Size tuning: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_size_tuning.csv`
- L23 recurrent specificity: `.runs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_full_5090_l23ee_specificity.csv`

Matrix/control logs:

- No-learning control: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_control_5090.log`
- SOM-off: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_somoff_5090.log`
- Recurrence-off: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_recoff_5090.log`
- PV-weak half-scale: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_pvweak_5090.log`
- PV-off: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_pvoff_5090.log`
- Strict matrix validator, half-PV: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_strict_matrix_validator_actual_pvweak.log`
- Strict matrix validator, PV-off causality: `.runs/logs/v1_ratehomeo_videol4scale085_truebaseline_pvrel105_pveta110_matrix_strict_matrix_validator_pvoff_for_pvcausality.log`

Rejected/diagnostic run logs:

- Whole-current divnorm beta 0.03: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm003_full_5090_validator_smoke.log`
- Whole-current divnorm beta 0.08: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm008_full_5090_validator_smoke.log`
- Whole-current divnorm beta 0.80: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm080_full_5090_validator_smoke.log`
- Floor-preserving divnorm: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm_floor080_full_5090_validator_smoke.log`
- Floor-preserving lagged divnorm beta 0.80: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm_floor080_lag_full_5090_validator_smoke.log`
- Floor-preserving lagged divnorm beta 0.20: `.runs/logs/v1_ratehomeo_videol4scale085_divnorm_floor020_lag_full_5090_validator_smoke.log`
- L4E adaptation: `.runs/logs/v1_ratehomeo_videol4scale085_l4eadapt0005_full_5090_validator_smoke.log`
- Analytic scale diagnostic: `.runs/logs/v1_ratehomeo_videol4scale085_analyticscale090_full_5090_rerun_validator_smoke.log`
