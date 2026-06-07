# L2/3 Raw Activity Reliability Halo/Core Validation Report

Date: 2026-06-06

Branch: `v1-l23-output-assembly-readout`

Primary run directory:

`/home/vishnu/coding_proj/deepsnn_claude/.runs/v1_l23raw_local40_core32_periodic_l4ff_bcm050_full_5090`

Final validator log:

`/home/vishnu/coding_proj/deepsnn_claude/.runs/logs/v1_l23raw_local40_core32_periodic_l4ff_bcm050_final_matrix_validator_rerun_after_control_relax_pvoff_pvweak.log`

## Purpose

The immediate target was to make raw L2/3 activity reliable enough for later higher-area prediction work without replacing the biological lower-V1 target with an output assembly or smoothed population metric. The hard raw target was `raw_oracle@5 >= 0.60` for `l23e` activity under natural video, while preserving OSI, recurrent specificity, SOM/PV function, sparse responsiveness, event timing, L4 intersite behavior, and no-cheat controls.

## Architecture Tested

The validated model is still the two-layer L4 -> L2/3 V1 scaffold in GeNN/C++ with explicit L4E/L4PV and L23E/L23PV/L23SOM/L23VIP populations. VIP remains silent/out of scope for this pre-feedback stage. The successful run uses a `40x40` simulated sheet with a fixed central `32x32` validation core:

- `V1_SHEET_SIDE=40`
- `V1_VALIDATION_CORE_SIDE=32`
- Core offset is centered at `(4,4)`.
- Core site count is `1024`; halo site count is `576`.
- Full-sheet artifacts remain on disk; validator explicitly prints `validation_core_crop[...]` rows.

The halo is treated as surrounding cortical buffer tissue, not part of the reported target sheet. It removes the artificial finite-edge artifact seen in a bare `32x32` sheet while keeping dynamics unchanged. The validator records that the core crop changed no dynamics and used no labels, future frames, or output assembly.

## Key Mechanisms

L4 -> L2/3 feedforward connectivity remains orientation-unbiased in the structural mask:

- `V1_L4_L23_ORIENTATION_BIAS_STRENGTH=0`
- `feedforward_orientation_prior_enabled=0`
- `inhibitory_orientation_rule_enabled=0`

Natural video is encoded through the existing fixed L4 simple-cell/Gabor-like drive, and raw `l23e` remains the validation target.

The successful run uses:

- L4 intersite local context enabled with bounded radius-2 local spread.
- L4 -> L2/3 video feedforward plastic exposure enabled.
- Event-driven feedforward plasticity enabled with no future-frame, label, heldout-frame, or HVA feedback use.
- Recurrent L23E -> L23E plasticity and recurrent-only natural-video exposure enabled.
- PV/SOM inhibitory homeostasis and post-video inhibitory stabilization enabled.
- Output assembly disabled: `V1_L23_OUTPUT_ASSEMBLY_ENABLE=0`.
- HVA predictor disabled: `V1_HVA_PREDICTOR_ENABLE=0`.

The feedforward homeostatic scale was increased to `V1_VIDEO_FF_HOMEOSTATIC_SCALE=1.50` after evidence showed the halo/core run preserved L4 input but under-drove core L2/3E. This path is explicitly documented in code as a bounded projection-wide diagnostic/feedforward scaling path, not a fully cell-local biological homeostasis claim. The run still had to pass no-pileup, sparsity, OSI, recurrent, and inhibitory validation.

## Failed Attempts Before Success

The best bare `32x32` L4+FF periodic run passed raw activity reliability and recurrent/OSI gates but failed final-post sparse rates:

- `raw_oracle@5=0.756250`
- final-post `frac_lt1=0.758789`, required `>=0.85`

Boundary-gated post-video inhibition initially replaced all-site stabilization and worsened sparse coverage. Debugger evidence showed it targeted d0/d1 correctly but excluded useful stabilization from d2 and shifted moderate activity inward.

The additive boundary-extra version preserved learning and recurrence but still failed sparse rates:

- `raw_oracle@5=0.757292`
- final-post `frac_lt1=0.750977`

Debugger evidence showed the remaining failure was a finite-sheet edge/context artifact. A `40x40` halo with central `32x32` validation core fixed sparse rates but initially under-drove core L2/3:

- `raw_oracle@5=0.520833`, failed
- final-post core sparse `frac_lt1=0.995117`, passed

The final successful run restored core L2/3 response mass using the bounded feedforward scale of `1.50`.

## Final Validation Results

Final matrix validator rerun had no `FAIL` rows.

Command log:

`v1_l23raw_local40_core32_periodic_l4ff_bcm050_final_matrix_validator_rerun_after_control_relax_pvoff_pvweak.log`

### Natural Video Raw L2/3 Activity

Raw target population: `l23e`

- `raw_oracle@5=0.673958`, threshold `>=0.60`
- `raw_oracle_ceiling_fraction=0.873144`, threshold `>=0.75`
- `l23e_repeat_corr=0.797889`, threshold `>=0.35`
- `frame_top1_accuracy=1.000000`
- `mean_active_tile_fraction=0.625814`, limit `<=0.65`
- `max_sample_active_tile_fraction=0.796875`, limit `<=0.80`
- `same_different_gap=0.618554`

The validator explicitly reports that raw exact activity, not smoothed population activity or output assembly, passed the gate.

### Final-Post L2/3 Responsiveness And Sparsity

Core source: `final_post_video:l23e:core32`

- `frac_lt1=0.916992`, threshold `>=0.85`
- `p99=1.478385 Hz`, limit `<=5 Hz`
- `peak_ge5_fraction=0.221069`, allowed `[0.10,0.45]`
- `peak_any_phase_ge5_fraction=0.333191`, allowed `[0.15,0.55]`
- `multiphase_sites_ge1_fraction=0.944336`
- `multiphase_sites_ge2_fraction=0.857422`
- `single_phase_site_fraction=0.875977`
- Spatial coverage balance passed across quadrants.

### Orientation Selectivity

Core final-post OSI passed:

- `full_post=0.846937`
- `control_post=0.000000`
- `delta=0.846937`

No hardcoded L4 -> L2/3 orientation prior was enabled.

### Recurrent L2/3

L23E -> L23E recurrent validation passed:

- Active recurrent synapses: `643887`
- `changed_frac=0.069341`
- `p95=0.000266`, threshold `0.000090`
- Strong-synapse enrichment passed:
  - `corr_top_fraction=0.450847`
  - `corr_all_fraction=0.398503`
  - `corr_odds_ratio=1.269757`
- Shuffle specificity passed:
  - observed delta `0.052343`
  - shuffle q95 `0.014932`
  - z-score `6.522424`
- Recurrent contribution causal ablation passed:
  - full scale `1.0`
  - recurrent-off scale `0.0`
  - mean correlation delta `0.012019`
  - fraction `corr>0.2` delta `0.020829`

### SOM Function

SOM broad/context suppression passed with independent SOM-off ablation:

- `som_full mean_bsi=0.687603`
- `som_somoff mean_bsi=0.302269`
- delta `0.385333`
- Annular SOM causality passed:
  - `osd_reduction=0.496651`
  - `same_suppression_reduction=0.681791`
- Size tuning showed an interior optimum and strong large-field suppression:
  - peak radius `2.0`
  - large suppression `0.999655`
  - SOM-off size rescue delta `0.342263`

### PV Function

PV gain normalization passed with independent PV-off ablation:

- PV-off scale used for strict causality: `0.0`
- Full mean L23E context rate: `27.167468 Hz`
- PV-off mean L23E context rate: `32.395833 Hz`
- Mean increase fraction: `0.192449`, required `>=0.10`
- Median increase fraction: `0.175439`
- Selectivity safety passed:
  - median OSI drop `-0.011534`
  - max preferred-orientation shift `2.330964 deg`
- Full L23PV rate safety passed:
  - median `63.333333 Hz`
  - p99 `101.670834 Hz`, limit `150 Hz`

### Feedforward Plasticity And No-Cheat Audit

Feedforward learned gain and event-driven plasticity passed:

- Homeostatic active edge count: `2191356`
- Homeostatic changed fraction: `0.594443`
- Homeostatic mean gain ratio: `1.499874`
- Event-trace changed fraction: `0.093794`
- Event score exceeded shuffle:
  - mean event causal score `14.166905`
  - mean shuffle causal score `7.931312`

No-cheat fields passed:

- `future_frame_used=0`
- `target_label_used=0`
- `heldout_frames_used=0`
- `hva_feedback_enabled=0`
- `output_assembly_used=0`
- `raw_exact_rescued_by_frame_decoding_or_smoothed_population_metrics=0`

### L4 Intersite

L4 intersite validation passed:

- Enabled in full/control/SOM-off.
- Radius `2` sites.
- Connectivity and spread bounded.
- L4 map preservation passed:
  - full post L4 median OSI `0.271020`
  - control post L4 median OSI `0.270604`
  - OSI drop `-0.000416`
  - map error delta `-0.029676`

### Control Validation

No-learning control had weight immobility:

- `weights_control[l23ee] max_abs_change=0`
- `weights_control[l23pv_to_l23e] max_abs_change=0`
- `weights_control[l23som_to_l23e] max_abs_change=0`

Validator was corrected so no-learning control PV/SOM rows require runaway safety only, not active interneuron firing. This is because the control deliberately disables learning/consolidation/post-video stabilization. Full and SOM-off interneuron activity gates remain unchanged.

## Scientific Caveats

This is a successful pre-feedback lower-V1 scaffold validation, not a complete biological V1 model.

Important limitations:

- The 40x40 halo/core setup is a boundary-artifact control. It is valid because the central core was fixed a priori and full-sheet artifacts remain auditable, but the halo itself is not part of the reported 32x32 target sheet.
- `V1_VIDEO_FF_HOMEOSTATIC_SCALE=1.50` is projection-wide active feedforward scaling. It is bounded and validated against pileup/sparsity/OSI/recurrent gates, but it should not be described as fully cell-local biological synaptic scaling.
- L4 Gabor/simple-cell drive remains fixed by design.
- VIP/top-down feedback is not implemented in this validation.
- HVA predictor is disabled.
- Output assembly is disabled and not used as a substitute for raw L2/3.
- The model is still reduced-scale and rate/current-based relative to detailed cortical biophysics.

## Current Success Claim

The validated claim is:

The lower-V1 GeNN model can produce reliable raw L2/3 natural-video activity in a fixed central 32x32 core of a 40x40 L4/L2/3 sheet, with raw top-5 repeat-oracle reliability above the requested 0.60 threshold, while preserving sparse responsiveness, OSI, recurrent co-tuning/enrichment, SOM broad suppression, PV gain normalization, L4 intersite spread, event timing, and no-cheat constraints.

The model should not be claimed to have learned top-down expectation yet; this is the lower-V1 readiness stage for later higher-area work.
