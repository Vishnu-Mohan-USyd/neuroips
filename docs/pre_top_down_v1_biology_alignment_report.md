# Pre-Top-Down V1 Biology Alignment Report

Branch snapshot: `v1-realistic-grating-training`

Primary H200 artifact directory:

```text
/scratch/proj/v1_snn_l4_l23/genn
```

This report summarizes the completed pre-top-down biology-alignment pass for
the reduced GeNN/C++ L4/L2/3 V1 model. It documents what is currently validated,
what is still hardcoded, what can be treated as emergent within this reduced
model, and what remains out of scope before adding top-down/VIP expectation
mechanisms.

## Current Validated Configuration

The latest passing configuration family is `v1_round7_sensory1_repeat_*`, a
Round 7 sensory-assay repeat of the `v1_pvnorm1_*` pre-top-down configuration.
It is a `40 x 40` sheet run with distributed validation, strict zero
L4-to-L2/3 orientation structural bias, and validation-only blank/contrast/
annular sensory assays.

Active compile/runtime controls:

```text
CXXFLAGS=-DV1_SHEET_SIDE=40
V1_VALIDATION_GRID_SIDE=3
V1_L4_INTERSITE_ENABLE=1
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27
V1_CELL_COVERAGE_PHASE_COUNT=4
V1_TRAINING_EPOCHS=2
V1_TRAINING_GRATING_MODE=phase_drift
V1_TRAINING_DRIFT_PHASE_COUNT=4
V1_L4_L23_ORIENTATION_BIAS_STRENGTH=0
V1_L4_L23_ORIENTATION_NEUTRAL_DENSITY_MATCH=1
V1_L4_L23_ORIENTATION_NEUTRAL_PROBABILITY_SCALE=1.27
V1_ORIENTATION_CONTEXT_ASSAY_ENABLE=1
V1_L23EE_LOGNORMAL_INIT=1
V1_L23EE_LOGNORMAL_SIGMA=0.37
V1_SENSORY_ASSAY_ENABLE=1
V1_BLANK_REPEAT_COUNT=4
V1_CONTRAST_SWEEP_VALUES=0.5,1.0
```

Run prefixes:

```text
v1_round7_sensory1_full
v1_round7_sensory1_control
v1_round7_sensory1_somoff
v1_round7_sensory1_recoff
v1_round7_sensory1_pvoff
v1_round7_sensory1_repeat_full
v1_round7_sensory1_repeat_control
v1_round7_sensory1_repeat_somoff
v1_round7_sensory1_repeat_recoff
v1_round7_sensory1_repeat_pvoff
v1_pvnorm1_full
v1_pvnorm1_control
v1_pvnorm1_somoff
v1_pvnorm1_recoff
v1_pvnorm1_pvweak
v1_pvnorm1_pvoff
```

Validator logs used for the final evidence:

```text
v1_pvnorm1_pvoff_validator.log
v1_pvnorm1_round4_som_validator_clean.log
v1_pvnorm1_round5_respsparse_validator.log
v1_pvnorm1_round6_scalingmap_validator.log
v1_round7_sensory1_validator.log
v1_round7_sensory1_repeat_validator.log
```

All model runs and final additive validators above returned `rc=0`. The earlier
`v1_pvnorm1_validator.log` used PV half-scale (`V1_L23PV_CONTEXT_OUTPUT_SCALE=0.5`)
and failed only the PV causality magnitude gate; the PV-off stress run
(`v1_pvnorm1_pvoff`, repeated as `v1_round7_sensory1_repeat_pvoff`) is the
passing PV causality artifact.

## Architecture

The model is a reduced two-sheet V1 scaffold:

- L4 is the driven input sheet.
- L2/3 is an aligned retinotopic cortical sheet.
- The validated sheet has `40 x 40 = 1600` sites.
- Per site: `16` L4E, `3` L4PV, `1` L4SOM, `16` L23E, `2` L23PV, `1` L23SOM,
  and `1` L23VIP.
- Total validated population size is `64,000` neurons.
- L4 inter-site spread is enabled as weak local static spread, not long-range
  patchy horizontal connectivity.

Validated L4 inter-site settings:

```text
radius_sites=2
weight_scale=0.12
l4ee_scale=0.06
l4e_to_l4pv_scale=0.03
l4pv_to_l4e_scale=0.03
max_projection_distance_sites=2.828427
max_same_site_fraction=0
max_beyond_radius_fraction=0
```

L4IS edge counts:

```text
l4ee_edges=9225216
l4e_to_l4pv_edges=1729728
l4pv_to_l4e_edges=1729728
```

## Hardcoded vs Emergent Ledger

Hardcoded by design:

- L4 retinotopic sheet geometry.
- L4 orientation map and phase-dependent grating drive.
- Population identities and broad connection motifs: E, PV, SOM, VIP.
- Reduced cell counts and site layout.
- Validation stimuli and held-out assay definitions.

Allowed hardcoded sensory drive:

- L4 orientation drive is allowed in this pre-top-down scaffold. The current
  model does not claim emergent L4 orientation maps or pinwheels.

Removed in strict validated configuration:

- L4E-to-L23E orientation structural prior is disabled:
  `l4_l23_orientation_bias_strength=0`.
- Feedforward orientation prior metadata is disabled:
  `l4_l23_feedforward_orientation_prior_enabled=0`.
- Inhibitory orientation rules are absent:
  `inhibitory_orientation_rule_enabled=0`.

Orientation-neutral density compensation:

- Enabled only to recover strict zero-bias L4-to-L23E convergence.
- Uses no orientation labels or similarity.
- Latest metadata: `l4_l23_edge_count=2123055`,
  `l4_l23_weights_before_nonzero_fraction=0.575916`,
  `l4_l23_weights_before_mean_all_slots=0.003455`.

Emergent/measured within this reduced model:

- Post-training L23E OSI.
- L2/3 orientation-context suppression under no L4-to-L23 orientation prior.
- L23E recurrent strong-weight heavy-tail-like distribution and response
  correlation enrichment.
- SOM-dependent size/surround suppression.
- PV gain-normalization causality under validation-only PV-off stress.
- True zero-drive blank silence and basic contrast monotonicity under
  validation-only sensory assays.
- Cell-level responsive coverage under held-out multiphase measurement.
- Distributed map consistency over `40 x 40` sheet.

Caveat: fixed local circuit dynamics can already produce some
orientation-context suppression in the no-learning control. The validator now
treats full-control OSD delta as plasticity enhancement/safety information, not
as proof that suppression is absent before plasticity.

## Training Protocol

The validated protocol uses opt-in phase-stepped grating training:

- `V1_TRAINING_GRATING_MODE=phase_drift`
- `V1_TRAINING_DRIFT_PHASE_COUNT=4`
- Each `250 ms` orientation trial is split into four `62.5 ms` phase slots.
- Phases are `0`, `90`, `180`, and `270` degrees.
- Traces and membrane states remain continuous across phase slots.
- Metadata reports `training_grating_counterbalance_enabled=1`.

Training/consolidation:

- Main plastic training epochs: `V1_TRAINING_EPOCHS=2`.
- Recurrent-only consolidation: `V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27`.
- Held-out post/context/size/recurrence/PV/SOM/scaling assays run with
  plasticity off.
- Round 7 blank/contrast/annular sensory assays are validation-only and run
  after training and context output-scale application.

## Validation Summary

### Round 1: Strict No-Hardcode and Orientation Context

Strict metadata passed:

```text
PASS no_hardcode_audit l4_l23_orientation_bias_strength=0.000000 feedforward_orientation_prior_enabled=0.000000 inhibitory_orientation_rule_enabled=0.000000
PASS l23_orientation_context_assay_enabled orientation_context_assay_enabled=1.000000
```

L23E orientation selectivity and orientation-context suppression:

```text
PASS osi full_post=0.822274 control_post=0.000000 delta=0.822274
PASS l23_orientation_context_driven_sites driven=9 required=5 site_count=9 threshold_hz=1.000000 mean_center_l23e_hz=42.118056
PASS l23_orientation_context_same_suppression mean_si_same_l23e=0.979958 median_si_same_l23e=0.993103
PASS l23_orientation_context_osd mean_osd_l23e=0.720574 median_osd_l23e=0.724138 control_mean_osd_l23e=0.668908
PASS l23_orientation_context_osd_site_fraction frac_osd_gt_0p05=1.000000
PASS l23_orientation_context_l23_minus_l4 mean_osd_l23e=0.720574 mean_osd_l4e=0.037614 delta=0.682960
PASS l23_orientation_context_full_control_delta full_mean_osd_l23e=0.720574 control_mean_osd_l23e=0.668908 delta=0.051665 plasticity_enhancement_informational=1 minimum_safety_delta=-0.020000
INFO l23_orientation_context_somoff mean_osd_l23e=0.192156 mean_si_same_l23e=0.281146 driven=9
```

Interpretation: L2/3 orientation-context suppression is strong in the strict
zero-bias configuration and substantially exceeds L4 OSD. The no-learning
control also has strong OSD, so this is evidence of emergent fixed-circuit plus
plastic dynamics, not plasticity-exclusive suppression.

### Round 2: Recurrent Biology

L23E-to-L23E recurrent initialization is orientation-blind lognormal:

```text
l23ee_lognormal_init_enabled=1
l23ee_lognormal_init_sigma=0.370000
l23ee_lognormal_init_target_mean=0.004500
l23ee_initial_active_count=643887
l23ee_initial_active_mean=0.004490
l23ee_initial_active_gini=0.203888
l23ee_initial_top10_mass_share=0.178506
```

Final recurrent distribution and specificity:

```text
PASS l23ee_recurrent_heavy_tail active_count=643887 p50=0.004210 p90=0.006772 p95=0.007750 p99=0.009901 max=0.010000 mean=0.004493 std=0.001675 cv=0.372882 gini=0.204707 top1pct_mass_share=0.022242 top5pct_mass_share=0.098483 top10pct_mass_share=0.178665 upper_cap_fraction=0.008519
PASS l23ee_recurrent_shuffle_specificity observed_top_fraction=0.343704 observed_all_fraction=0.305032 observed_delta=0.038673 shuffle_q95_delta=0.016067 z_score=5.267976
PASS l23ee_recurrent_cotuning_bins high_mean_w_after=0.004555 low_mean_w_after=0.004492 high_mean_delta_w=0.000057 low_mean_delta_w=0.000002 best_margin=0.000073
PASS l23ee_strong_synapse_enrichment corr_odds_ratio=1.217885 combined_odds_ratio=1.303890
```

Recurrence causal contribution:

```text
PASS l23ee_recurrence_corr_contribution mapped_pairs=643887 active_pairs=222543 focus_pairs=66417 mean_corr_on=0.114867 mean_corr_off=0.089892 mean_corr_delta=0.024975 frac_corr_gt_0p2_on=0.187377 frac_corr_gt_0p2_off=0.150669 frac_corr_gt_0p2_delta=0.036707
PASS l23ee_recurrence_rate_osi_safety mean_peak_on=5.393890 mean_peak_off=4.923543 peak_ratio_off_on=0.912800 mean_osi_on=0.532215 mean_osi_off=0.449087
```

Interpretation: recurrent weights are bounded-heavy-tail-like, response
correlation enrichment exceeds the distance-preserving shuffled null, and
validation-only recurrence removal reduces connected-pair response correlation
without catastrophic selectivity/rate loss.

### Round 3: PV Gain Normalization

PV half-scale was too weak to satisfy the causality magnitude gate:

```text
FAIL pv_gain_normalization_causality pvweak_scale=0.500000 mean_increase_fraction=0.068849 median_increase_fraction=0.075269 required_gain_floor=0.200000
```

The validated PV causality stress uses PV-off context output scaling:

```text
PASS pv_gain_normalization_causality pvweak_scale=0.000000 pvweak_active=1 site_count=9 driven_rate_count=81 full_mean_l23e_hz=29.587191 pvweak_mean_l23e_hz=33.445216 mean_increase_fraction=0.130395 full_median_l23e_hz=29.062500 pvweak_median_l23e_hz=33.125000 median_increase_fraction=0.139785 required_gain_floor=0.100000 pvweak_l23e_context_p99_hz=57.000000 p99_limit_hz=100.000000
PASS pv_gain_normalization_selectivity_safety full_median_osi=0.405260 pvweak_median_osi=0.435398 median_osi_drop=-0.030139 median_pref_shift_deg=0.798096 max_pref_shift_deg=1.759101
PASS pv_gain_normalization_rates full_l23pv_post_median_hz=38.750000 full_l23pv_post_frac_lt1=0.056250 full_l23pv_post_p99_hz=67.925000 p99_limit_hz=150.000000
```

Interpretation: PV output is active, bounded, and causally suppresses L23E gain
under validation-only PV-off stress. The half-scale assay remains informative
but not strong enough as a required perturbation.

### Round 4: SOM Size/Surround

SOM and size tuning passed:

```text
PASS som_full mean_bsi=0.900867 driven_center_threshold_hz=11.197917 validation_sites=9 relevant_orientations=81
PASS som_somoff full_mean_bsi=0.900867 somoff_mean_bsi=0.243809 delta=0.657058
PASS som_size_curve_shape validation_sites=9 peak_radius=2.000000 small_l23e_rate=0.000000 peak_l23e_rate=41.736111 large_l23e_rate=0.000000 summation_index=1.000000 large_suppression_index=1.000000 site_curve_pass_fraction=1.000000
PASS som_size_l4_vs_l23e l23e_suppression=1.000000 l4e_suppression=0.068852 delta=0.931148 threshold=0.050000
PASS som_size_som_recruitment peak_som_rate=32.592593 large_som_rate=12.592593 center_context_som_rate=33.888889 broad_context_som_rate=27.777778 som_recruitment_index=-0.180328
PASS som_size_somoff_site_rescue site_rescue_fraction=1.000000 somoff_large_l23e_rate=34.699074 mean_suppression_reduction_full_minus_somoff=0.372097
PASS som_orientation_context_somoff_effect full_mean_osd_l23e=0.720574 somoff_mean_osd_l23e=0.192156 osd_reduction=0.528418 same_suppression_reduction=0.698812
```

Interpretation: L23E has a driven, peaked size curve with large-radius
suppression; L23E suppression substantially exceeds L4 suppression; SOM remains
active at broad/large contexts; and SOM-off rescues large-radius/parallel
orientation-context L23E response.

### Round 5: Responsiveness and Sparsity

Cell-level sparse responsiveness passed:

```text
PASS responsiveness_artifacts_available full_cell_tuning=1 full_multiphase_tuning=1 missing=none
PASS l23e_cell_sparse_responsiveness peak_ge5_fraction=0.208437 peak_ge10_fraction=0.112617 peak_ge10_cells=2883 total_cells=25600
PASS l23e_cell_multiphase_sparse_responsiveness peak_any_phase_ge5_fraction=0.337070 peak_any_phase_ge10_fraction=0.225000 responsive_median_phase_pooled_osi=0.892820 total_cells=25600
PASS l23e_responsive_site_coverage multiphase_sites_ge1_fraction=0.925625 multiphase_sites_ge2_fraction=0.798750 single_phase_site_fraction=0.763125 total_multiphase_sites=1600
PASS l23e_population_sparse_rates frac_lt1=0.946250 p99=1.719010 p99_limit=5.000000
PASS l23e_spatial_coverage_balance responsive_sites=1481 responsive_cells=8629 min_quadrant_site_fraction=0.246455 min_quadrant_cell_fraction=0.244293 zero_site_quadrants=0 zero_cell_quadrants=0
INFO blank_spontaneous_baseline_missing artifact_available=0 hard_fail=0
```

Interpretation: L23E is sparse at the population/site-rate level while retaining
substantial cell-level responsiveness under multiphase held-out measurement.
Round 7 adds the missing true zero-drive blank baseline; Round 5 values above
remain the cell responsiveness/sparsity evidence.

### Round 6: Scaling and Map Consistency

Distributed map/scaling checks passed:

```text
PASS scaling_map_artifacts_available missing=none
PASS scaling_l4_map_consistency active_sites=1600 total_sites=1600 active_fraction=1.000000 median_map_error_deg=0.779173 p90_map_error_deg=2.005588
PASS scaling_l23_l4_map_consistency active_site_count=1221 active_site_median_delta_deg=21.885082 active_site_p90_delta_deg=74.537815 cell5_count=8629 cell5_median_delta_deg=19.597825 cell10_count=5760 cell10_median_delta_deg=15.631152
PASS scaling_tile_orientation_coverage responsive_cells=8629 nonempty_tiles=16 global_occupied_bins=12/12 min_tile_cell_count=469 max_tile_cell_count=686 min_occupied_bins=12 median_occupied_bins=12.000000
PASS scaling_tile_orientation_entropy threshold5_median_entropy=0.942552 threshold5_min_entropy=0.876109 threshold10_median_entropy=0.862835 threshold10_min_entropy=0.784993
PASS scaling_edge_quadrant_balance responsive_edge_sites=135 edge_sites=156 edge_site_coverage=0.865385 min_quadrant_cell_fraction=0.244293 zero_quadrants=0
PASS scaling_l4is_preservation enabled_ok=1 connectivity_ok=1 map_ok=1 spread_ok=1 radius_sites=2.000000 max_projection_scale=0.060000 post_l4_map_error_deg=0.779173 map_error_delta=0.000000 l4_peak_rate_hz=156.863426 small_peak_ratio=0.526009 large_peak_ratio=0.884896
```

Interpretation: L4 map readout is stable across all active sites; L23 active
sites and responsive cells remain broadly aligned to L4; all 4x4 tiles contain
responsive cells and all 12 orientation bins; edge/quadrant coverage does not
collapse.

### Round 7: Sensory Baseline, Contrast, and Annular Surround

Round 7 used validation-only sensory assays in `v1_round7_sensory1_repeat`.
The blank assay explicitly sets all L4E external current to zero; it does not
use contrast `0`, because the L4 drive function has baseline bias terms.

Blank baseline passed with all site/population p99 rates at zero:

```text
PASS sensory_blank_artifacts_available sensory_assay_enabled=1.000000 blank_rows=25600 missing=none
INFO sensory_blank_summary repeat_count=4 l4e_p50=0.000000 l4e_p95=0.000000 l4e_p99=0.000000 l23e_p50=0.000000 l23e_p95=0.000000 l23e_p99=0.000000 l23pv_p99=0.000000 l23som_p99=0.000000
PASS sensory_blank_l4_low repeat_count=4 l4e_mean_hz=0.000000 l4e_p99_hz=0.000000 l4e_max_hz=0.000000
PASS sensory_blank_l23e_sparse_safe l23e_frac_lt1=1.000000 l23e_p50_hz=0.000000 l23e_p95_hz=0.000000 l23e_p99_hz=0.000000
PASS sensory_blank_interneuron_safe l23pv_p50_hz=0.000000 l23pv_p99_hz=0.000000 l23som_p50_hz=0.000000 l23som_p99_hz=0.000000
```

Contrast sweep passed for `0.5` and `1.0` contrast at validation sites:

```text
PASS sensory_contrast_artifacts_available sensory_assay_enabled=1.000000 contrast_rows=72 missing=none
PASS sensory_contrast_l4_monotonic low_contrast=0.500000 high_contrast=1.000000 site_count=9 l4e_low_mean_hz=194.166667 l4e_high_mean_hz=237.500000 l4e_mean_delta_hz=43.333333 l4e_monotonic_fraction=1.000000
PASS sensory_contrast_l23e_gain_safe site_count=9 l23e_low_mean_hz=31.145833 l23e_high_mean_hz=41.666667 required_delta_hz=0.622917 l23e_mean_delta_hz=10.520833 l23e_monotonic_fraction=1.000000 l23e_high_p99_hz=46.200000 p99_limit_hz=100.000000
```

True annular surround protocol passed: surround-only rows used
`inner_radius_sites=2` and `outer-inner=1`, same-vs-orthogonal OSD was strong,
and SOM-off reduced annular OSD/same suppression:

```text
PASS sensory_annular_artifacts_available full_rows=45 somoff_rows=45 full_missing=0 somoff_missing=0
PASS sensory_annular_protocol_present orientation_context_assay_enabled=1.000000 annular_surround_only_enabled=1.000000 surround_only_rows=18 annular_row_fraction=1.000000 min_inner_radius_sites=2.000000 min_outer_minus_inner_sites=1.000000
PASS sensory_annular_surround_only_low driven=9 required_driven=5 mean_center_l23e_hz=41.770833 mean_surround_only_l23e_hz=0.000000 mean_surround_only_ratio=0.000000 rate_guard_hz=10.442708
PASS sensory_annular_same_vs_orth_osd driven=9 required_driven=5 mean_si_same_l23e=0.981330 median_si_same_l23e=1.000000 mean_osd_l23e=0.727238 median_osd_l23e=0.733333 mean_osd_l4e=0.037614
PASS sensory_annular_som_causality full_driven=9 somoff_driven=9 full_mean_osd_l23e=0.727238 somoff_mean_osd_l23e=0.191982 osd_reduction=0.535257 full_mean_si_same_l23e=0.981330 somoff_mean_si_same_l23e=0.281680 same_suppression_reduction=0.699649
```

Interpretation: the model is silent under true zero L4 drive, responds
monotonically to the minimal contrast sweep without exceeding rate safety
limits, and preserves SOM-dependent annular same-vs-orthogonal surround
suppression.

### Repeat Robustness and Determinism

There is no explicit seed environment variable in the current model interface.
The local sparse topology and lognormal L23E-to-L23E initialization use
deterministic hash functions over cell/edge identities, so a same-config repeat
primarily checks build/runtime robustness rather than multi-seed variability.

The full repeat `v1_round7_sensory1_repeat` passed with stable key margins
relative to `v1_round7_sensory1`:

```text
osi.full_post: original=0.809028 repeat=0.815859 delta=+0.006831
heavy_tail.gini: original=0.204698 repeat=0.204712 delta=+0.000014
pv.mean_increase_fraction: original=0.132612 repeat=0.132709 delta=+0.000097
pv.p99: original=56.562500 repeat=56.562500 delta=+0.000000
annular.osd_reduction: original=0.549878 repeat=0.535257 delta=-0.014621
annular.same_suppression_reduction: original=0.702144 repeat=0.699649 delta=-0.002495
blank.l23e_p99: original=0.000000 repeat=0.000000 delta=+0.000000
blank.l23pv_p99: original=0.000000 repeat=0.000000 delta=+0.000000
contrast.l4_delta: original=43.333333 repeat=43.333333 delta=+0.000000
contrast.l23e_delta: original=9.826389 repeat=10.520833 delta=+0.694444
contrast.l23e_high_p99: original=47.087500 repeat=46.200000 delta=-0.887500
```

This is not a multi-seed robustness claim; it is evidence that the full Round 7
pipeline and validator are repeatable under the current deterministic run
construction.

## Mismatches Found and Fixes Applied

### Zero-bias density collapse

Problem: setting `V1_L4_L23_ORIENTATION_BIAS_STRENGTH=0` removed the structural
orientation prior but also reduced L4-to-L23E density/drive enough to collapse
L23 coverage.

Fix: add opt-in orientation-neutral density matching:

```text
V1_L4_L23_ORIENTATION_NEUTRAL_DENSITY_MATCH=1
V1_L4_L23_ORIENTATION_NEUTRAL_PROBABILITY_SCALE=1.27
```

Latest strict run has `l4_l23_weights_before_nonzero_fraction=0.575916`.
Because this uses no orientation labels/preferences/similarity, it preserves the
strict no-hardcode interpretation.

### Uniform recurrent weights

Problem: L23E-to-L23E recurrent weights initialized nearly uniformly and additive
STDP left most weights unchanged, failing bounded-heavy-tail biology checks.

Fix: opt-in orientation-blind lognormal initialization:

```text
V1_L23EE_LOGNORMAL_INIT=1
V1_L23EE_LOGNORMAL_SIGMA=0.37
```

Latest metadata: initial active Gini `0.203888`, initial top10 mass share
`0.178506`; final active Gini `0.204707`, top10 mass share `0.178665`.

### PV half-scale perturbation too weak

Problem: `V1_L23PV_CONTEXT_OUTPUT_SCALE=0.5` increased driven L23E rates by only
`6.9%` mean and `7.5%` median, below the Round 3 gate.

Fix/interpretation: keep PV half-scale as an informative weak perturbation, but
validate PV gain causality with PV-off stress:

```text
V1_L23PV_CONTEXT_OUTPUT_SCALE=0
```

The PV-off validator passed with mean L23E increase `0.130395`, median increase
`0.139785`, preserved selectivity, and bounded p99.

### Full-control OSD interpretation

Problem: no-learning control had strong orientation-context OSD
(`control_mean_osd_l23e=0.668908`), so a strict full-control delta gate
conflated plasticity-specific enhancement with fixed local circuit emergence.

Fix: keep reporting full/control OSD and require only a safety/non-negative
delta under the no-hardcode assay. The latest run still has positive plasticity
enhancement (`delta=0.051665`), but the biological claim is not that control OSD
is absent.

## Limitations and Out of Scope

Not implemented:

- No true LGN spiking afferent model.
- No conductance synapses or dendritic compartments.
- No multi-seed validation yet, because the model currently has no explicit
  seed/salt mechanism for topology or lognormal initialization.
- No natural image or natural movie stimulus battery.
- Blank baseline is site/population-rate only, not cell-level spontaneous
  activity characterization.
- No emergent L4 orientation map or pinwheel development.
- No top-down expectation pathway yet.
- No learned VIP expectation/disinhibitory control yet.
- No full-scale cortical density or anatomical hypercolumns.

Important simplifications:

- L4 orientation drive is allowed and hardcoded.
- `40 x 40` is still a reduced sheet, not biological V1 scale.
- Cell counts per site are low relative to real cortex.
- The L4IS mechanism is weak local spread, not long-range L2/3-like horizontal
  patchiness.
- Some validations are site-level or assay-level diagnostics, not direct
  biological proof.
- Orientation-context assays use controlled validation stimuli; the model does
  not yet implement a full naturalistic stimulus battery.
- Responsive coverage is measured with held-out phase sweeps, not a full
  contrast/spontaneous tuning protocol.

## Bottom Line

The current `v1_pvnorm1` / `v1_round7_sensory1` family is the first strict
pre-top-down configuration that simultaneously validates, with Round 7 repeated
as `v1_round7_sensory1_repeat`:

- `40 x 40` distributed sheet behavior.
- Phase-drift grating training.
- Zero L4-to-L23E orientation structural prior.
- Orientation-neutral L4-to-L23E density compensation.
- L23E recurrent bounded-heavy-tail-like initialization and correlation
  enrichment.
- SOM size/surround suppression and SOM-off rescue.
- PV gain-normalization causality under PV-off stress.
- True zero-drive blank silence, monotonic `0.5/1.0` contrast response, and
  true annular surround/SOM causality.
- Sparse but non-collapsed L23E cell responsiveness.
- L4/L23 map and tile coverage consistency.

The model is therefore ready for the next stage: adding top-down/VIP expectation
mechanisms without relying on a hardcoded L4-to-L23 orientation structural prior.
