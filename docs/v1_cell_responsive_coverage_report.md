# V1 Cell-Level Responsive Coverage Report

Branch: `v1-cell-responsive-coverage`

This report documents the correction of the scaled `40 x 40` L4 inter-site
model's responsive-coverage problem. It builds on
`docs/v1_l4_intersite_scaling_report.md`.

## 1. Problem

The previous `40 x 40` L4 inter-site run passed L4 inter-site, SOM/PV, rate,
recurrent specificity, recurrent causality, and responsive-site OSI gates, but
failed the strict all-site L2/3 OSI gate:

```text
FAIL osi full_post=0.685620 control_post=0.000000 delta=0.685620
```

The apparent responsive coverage was also low when measured from site-mean L2/3
activity:

```text
site responsive threshold=1 Hz
responsive sites=100/1600
responsive site fraction=0.062500
```

That site-level metric was not biologically equivalent to cell-level visual
responsiveness because each site averages `16` L23E cells and the old post
sweep used only one fixed spatial phase.

## 2. Measurement Fix

The model already has phase-dependent L4 simple-cell drive. L4E neurons are
split into four phase subtypes:

```text
0, pi/2, pi, 3pi/2
```

Training phase-cycles during plastic exposure, but the legacy post sweep used a
single non-plastic phase. This can undercount phase-selective L2/3 cells.

This branch adds an opt-in held-out multi-phase coverage sweep:

```text
V1_CELL_COVERAGE_PHASE_COUNT=4
```

When enabled, the trained network is frozen and measured over:

```text
orientation x phase
```

with no plasticity. The model writes:

```text
<prefix>_l23e_cell_tuning_multiphase.csv
```

The validator now reports:

```text
INFO l23e_cell_multiphase_coverage[...]
INFO l23e_cell_responsive_coverage[...]
```

No legacy output or strict gate is changed by default.

## 3. First Multi-Phase Measurement

Using the previous `40 x 40` L4 inter-site configuration with one training
epoch and four held-out phases:

```text
V1_CELL_COVERAGE_PHASE_COUNT=4
V1_L4_INTERSITE_ENABLE=1
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27
```

Coverage improved relative to single-phase measurement:

| Metric | Single phase | Multi-phase |
|---|---:|---:|
| Responsive cells at `>=5 Hz` | `0.173594` | `0.260625` |
| Sites with `>=1` responsive cell | `0.671250` | `0.845000` |
| Sites with `>=2` responsive cells | not measured | `0.665000` |
| Responsive median phase-pooled OSI | not measured | `0.868939` |

This showed that fixed-phase measurement was undercounting responsive cells.
However, `0.260625` still missed the target `>=0.30` peak-responsive cell
fraction.

## 4. Mechanism Fix

Debugger analysis showed:

- Coverage was plasticity-responsive: full `0.260625` vs control `0.104648`.
- The gap to `0.30` was modest: about `1008` additional L23E cells.
- L23E rates had large safety headroom.
- L4 inter-site gates passed.
- SOM was not limiting coverage.
- Feedforward weights were not saturated.

The smallest targeted mechanism was therefore one extra plastic training epoch:

```text
V1_TRAINING_EPOCHS=2
```

No topology, excitability, inhibition, L4 inter-site balance, or validator
threshold was changed for this mechanism fix.

## 5. Final Validated Configuration

The successful scaled configuration:

```text
-DV1_SHEET_SIDE=40
V1_VALIDATION_GRID_SIDE=3
V1_L4_INTERSITE_ENABLE=1
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27
V1_CELL_COVERAGE_PHASE_COUNT=4
V1_TRAINING_EPOCHS=2
```

H200 prefixes:

```text
v1_l4is40_grid3_train2_ro27_ei006003_mp4_full
v1_l4is40_grid3_train2_ro27_ei006003_mp4_control
v1_l4is40_grid3_train2_ro27_ei006003_mp4_somoff
v1_l4is40_grid3_train2_ro27_ei006003_mp4_recoff
```

Validator status:

```text
rc=0
```

## 6. Final Coverage Results

Cell-level multi-phase coverage at `>=5 Hz`:

```text
INFO l23e_cell_multiphase_coverage[full] available=1 threshold_hz=5.000000 total_cells=25600 active_cells=8248 active_fraction=0.322188 responsive_cells=8248 responsive_fraction=0.322188 responsive_median_phase_pooled_osi=0.896618 total_sites=1600 active_sites_ge1=1455 active_site_fraction_ge1=0.909375 responsive_sites_ge1=1455 responsive_site_fraction_ge1=0.909375 responsive_sites_ge2=1252 responsive_site_fraction_ge2=0.782500
INFO l23e_cell_multiphase_coverage[control] available=1 threshold_hz=5.000000 total_cells=25600 active_cells=2638 active_fraction=0.103047 responsive_cells=2638 responsive_fraction=0.103047 responsive_median_phase_pooled_osi=0.891905 total_sites=1600 active_sites_ge1=955 active_site_fraction_ge1=0.596875 responsive_sites_ge1=955 responsive_site_fraction_ge1=0.596875 responsive_sites_ge2=589 responsive_site_fraction_ge2=0.368125
INFO l23e_cell_multiphase_coverage[somoff] available=1 threshold_hz=5.000000 total_cells=25600 active_cells=8354 active_fraction=0.326328 responsive_cells=8354 responsive_fraction=0.326328 responsive_median_phase_pooled_osi=0.892255 total_sites=1600 active_sites_ge1=1474 active_site_fraction_ge1=0.921250 responsive_sites_ge1=1474 responsive_site_fraction_ge1=0.921250 responsive_sites_ge2=1265 responsive_site_fraction_ge2=0.790625
```

The full run now clears the `>=0.30` hard target:

```text
responsive_cells=8248/25600 = 0.322188
responsive_sites_ge1=1455/1600 = 0.909375
responsive_sites_ge2=1252/1600 = 0.782500
responsive_median_phase_pooled_osi=0.896618
```

## 7. Final Validation Results

The final run passes all strict gates:

```text
PASS validation_sites required=9 full_context=9 somoff_context=9 full_size=9 somoff_size=9
PASS l4_intersite_enabled full=1 control=1 somoff=1
PASS l4_intersite_connectivity radius_sites=2.000000 weight_scale=0.120000 l4ee_scale=0.060000 l4e_to_l4pv_scale=0.030000 l4pv_to_l4e_scale=0.030000 l4ee_edges=9225216 l4e_to_l4pv_edges=1729728 l4pv_to_l4e_edges=1729728 max_distance_sites=2.828427 max_same_site_fraction=0.000000 max_beyond_radius_fraction=0.000000
PASS l4_intersite_map_preservation full_post_l4_median_osi=0.270658 control_post_l4_median_osi=0.270658 osi_drop=0.000000 allowed_osi_drop=0.020000 full_post_l4_map_error_deg=0.791778 control_post_l4_map_error_deg=0.791778 map_error_delta=0.000000
PASS l4_intersite_spread_bounded l4_peak_rate_hz=156.863426 small_peak_ratio=0.525935 large_peak_ratio=0.884896
PASS osi full_post=0.779906 control_post=0.000000 delta=0.779906
PASS weights_full[l23ee] active=643887 changed_frac=0.077559 p95=0.000169 threshold=0.000090 lower_frac=0.000006 upper_frac=0.000002 min_nonzero=0.001000 max_nonzero=0.010000
PASS rates[full:l23e] median=0.078125 frac_lt1=0.955000 p99=1.745052 limit=100.0
PASS som_size_somoff full_suppression=1.000000 somoff_suppression=0.614589 delta=0.385411
PASS l23ee_specificity best_margin=0.000003
PASS l23ee_response_corr_specificity best_margin=0.000003
PASS l23ee_strong_synapse_enrichment corr_odds_ratio=2.585056 combined_odds_ratio=2.996527
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.028726 frac_corr_gt_0p2_delta=0.042027
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.527351 mean_osi_off=0.444904
PASS vip_weights none found
```

## 8. Biological Interpretation

The coverage issue is now substantially improved:

- Cell-level responsive coverage reaches `32.2%`.
- Site coverage with at least one responsive L23E reaches `90.9%`.
- Site coverage with at least two responsive L23E reaches `78.3%`.
- Responsive-cell phase-pooled OSI remains high.
- Existing SOM, PV, recurrent, L4 inter-site, and rate-safety gates remain
  intact.

This is still not full primate/cat density or full biological realism. The
model remains a reduced scaffold with hardcoded L4 orientation drive and a soft
orientation-biased L4 to L2/3 prior. However, the previous claim that coverage
was likely too sparse is no longer true for the validated multi-phase,
two-training-epoch configuration.

## 9. Remaining Caveats

Still not implemented:

- Emergent pinwheels.
- Emergent L4 orientation map.
- Top-down/VIP expectation feedback.
- True annular surround.
- Multi-seed robustness.
- Conductance synapses and dendritic compartments.

Still simplified:

- Multi-phase static sweeps approximate drifting-grating phase sampling.
- Coverage thresholds are biologically motivated but not yet calibrated against
  a matched experimental stimulus protocol.
- The model is still much smaller and more abstract than cat/macaque V1.

## 10. Bottom Line

The scaled `40 x 40` model with opt-in L4 inter-site connectivity now passes
strict OSI, L4, SOM, PV/rate, recurrent plasticity, recurrent specificity,
recurrence causality, and cell-level responsive coverage checks when measured
with a biologically fair multi-phase held-out protocol and two training epochs.
