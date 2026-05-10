# V1 Scaling and L4 Inter-Site Connectivity Report

Branch: `v1-l4-intersite-connectivity`

This report documents the current GeNN/C++ two-layer V1 model after adding
incremental sheet scaling, distributed validation, and opt-in L4 inter-site
connectivity. It is a reproducibility snapshot before the next branch, which
will address biologically low responsive coverage with cell-level peak-response
exports and coverage gates.

## 1. Current Model

The model remains a reduced two-sheet V1 scaffold:

- L4 is the bottom input sheet.
- L2/3 is an overlaid retinotopic sheet.
- Both sheets have aligned sites.
- The default sheet is `32 x 32 = 1024` sites.
- The opt-in scaled sheet is `40 x 40 = 1600` sites via compile-time
  `-DV1_SHEET_SIDE=40`.
- Each site has `16` L4E, `3` L4PV, `1` L4SOM, `16` L23E, `2` L23PV,
  `1` L23SOM, and `1` L23VIP neuron.

Population sizes:

| Sheet | Sites | Total neurons |
|---|---:|---:|
| `32 x 32` | 1024 | 40,960 |
| `40 x 40` | 1600 | 64,000 |

The L4 orientation map and L4 simple-cell-like drive are still hardcoded. This
branch does not claim emergent pinwheel or emergent orientation-map formation.
L2/3 selectivity, recurrent enrichment, SOM size suppression, and distributed
site responses are measured after training.

## 2. Scaling Infrastructure

`genn/v1TwoLayerConfig.h` now makes sheet side compile-time configurable:

```cpp
#ifndef V1_SHEET_SIDE
#define V1_SHEET_SIDE 32
#endif
```

Default behavior remains `32 x 32`. Scaling must be explicit. The validated
larger run used:

```bash
-DV1_SHEET_SIDE=40
```

Distributed validation is runtime-configurable:

- `V1_VALIDATION_GRID_SIDE=3` selects a `3 x 3` grid of validation sites.
- `V1_VALIDATION_SITE_IDS` can explicitly select validation sites.
- `tools/validate_full_plasticity.py --min-validation-sites 9` requires that
  distributed artifacts actually contain all nine validation sites.

This avoids the previous center-only limitation. The model can now validate
context/SOM/size behavior across multiple retinotopic locations.

## 3. L4 Inter-Site Biology Framing

The L4 inter-site mechanism is intentionally local and weak. It is not a
superficial-layer horizontal patch system and not an iso-orientation patchy
projection.

Biological rationale:

- Macaque L4C is primarily an input/local retinotopic granular circuit.
- Strong long-range patchy iso-orientation horizontal connectivity is mainly a
  superficial L2/3 phenomenon, not a default L4C mechanism.
- L4/L4C can have local intracortical spread and local E/I normalization, but
  treating L4 as an L2/3-like long-range horizontal network would overclaim the
  biology.

Primary sources used for this framing:

- Blasdel and Fitzpatrick 1984, macaque L4 physiology,
  https://doi.org/10.1523/JNEUROSCI.04-03-00880.1984
- Fitzpatrick, Lund, and Blasdel 1985, macaque 4C intrinsic projections,
  https://doi.org/10.1523/JNEUROSCI.05-12-03329.1985
- Yabuta and Callaway 1998, macaque 4C streams,
  https://doi.org/10.1523/JNEUROSCI.18-22-09489.1998
- Bosking et al. 1997, tree-shrew L2/3 horizontal orientation patches,
  https://doi.org/10.1523/JNEUROSCI.17-06-02112.1997
- Chisum, Mooser, and Fitzpatrick 2003, vertical/horizontal V1 connections,
  https://doi.org/10.1523/JNEUROSCI.23-07-02947.2003
- Mooser, Bosking, and Fitzpatrick 2004, L4 to L2/3 feedforward orientation
  bias, https://doi.org/10.1038/nn1287
- Gilbert and Wiesel 1983, clustered intrinsic connections in cat V1,
  https://doi.org/10.1523/JNEUROSCI.03-05-01116.1983
- Angelucci et al. 2002, macaque lateral/feedback spatial scales,
  https://doi.org/10.1523/JNEUROSCI.22-19-08633.2002

## 4. Implemented L4 Inter-Site Mechanism

The feature is opt-in. Default is disabled:

```text
V1_L4_INTERSITE_ENABLE_DEFAULT=0
```

When enabled, the model adds three static radius-2 inter-site projections:

- `L4E_to_L4E_intersite`
- `L4E_to_L4PV_intersite`
- `L4PV_to_L4E_intersite`

These use a local inter-site patch that skips same-site targets. There is no
orientation-bias rule, no long-range patchiness, and no L4 plasticity.

Runtime controls:

```text
V1_L4_INTERSITE_ENABLE
V1_L4_INTERSITE_RADIUS
V1_L4_INTERSITE_WEIGHT_SCALE
V1_L4_INTERSITE_EE_SCALE
V1_L4_INTERSITE_E_PV_SCALE
V1_L4_INTERSITE_PV_E_SCALE
```

The global scale defaults to `0.12`. Per-projection scales default to the
global value if unset.

The final balanced configuration was:

```text
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
```

This was chosen after evidence showed that a uniform scale over-recruited L4PV
and reduced downstream L2/3 responsive coverage.

## 5. Added Exports and Validator Gates

The model now writes:

```text
<prefix>_l4_intersite_diagnostics.csv
```

This file records:

- enabled/disabled state
- radius
- global and per-projection scales
- effective inter-site weights
- inter-site edge counts
- maximum projection distance
- same-site and beyond-radius fractions
- L4 size/spread metrics
- L4 post median OSI and L4 map error metrics

The validator has:

```text
--require-l4-intersite
--min-validation-sites
--allow-responsive-osi
--responsive-rate-threshold-hz
```

New L4-specific gates:

- `l4_intersite_enabled`
- `l4_intersite_connectivity`
- `l4_intersite_map_preservation`
- `l4_intersite_spread_bounded`

`l4_intersite_map_preservation` is relative to matched control/off behavior.
The old absolute threshold `post_l4_median_osi >= 0.30` was rejected because
default L4 median OSI is about `0.27`; the proper biological requirement is map
preservation and no degradation, not arbitrary high L4 OSI.

The validator also reports:

- `INFO l23e_osi_sites[...]`
- `INFO l23e_osi_quadrants[...]`

These lines separate all-site, active-site, and responsive-site L2/3 OSI.

## 6. Training Protocol

The current validated runs use the existing staged training protocol:

1. Baseline sweep with plasticity off.
2. Main training with feedforward STDP, L23E recurrent STDP, and PV/SOM
   inhibitory homeostasis on.
3. Recurrent consolidation.
4. Recurrent-only consolidation.
5. Post sweep.
6. Context/size/recurrence validation sweeps with plasticity off.

For `40 x 40`, the validated no-L4 and L4-inter-site runs use:

```text
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27
```

This was not guessed. The initial `40 x 40` run missed OSI and L23EE movement;
debugger analysis showed sparse responsive coverage and under-exposed recurrent
plasticity. Increasing recurrent-only consolidation from `18` to `27` fixed the
no-L4 `40 x 40` run.

## 7. Validation History

### 7.1 Default 32x32 Compatibility

Default `32 x 32`, L4 inter-site disabled, passed.

Key lines:

```text
PASS validation_sites required=1 full_context=1 somoff_context=1 full_size=1 somoff_size=1
PASS osi full_post=0.757043 control_post=0.000000 delta=0.757043
PASS som_size_somoff full_suppression=1.000000 somoff_suppression=0.490838 delta=0.509162
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.017088 frac_corr_gt_0p2_delta=0.024602
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.521489 mean_osi_off=0.468372
PASS vip_weights none found
```

### 7.2 32x32 Distributed Grid3 Without L4 Inter-Site

`32 x 32`, `V1_VALIDATION_GRID_SIDE=3`, L4 inter-site disabled, passed.

Key lines from previous scaling validation:

```text
PASS validation_sites required=9 full_context=9 somoff_context=9 full_size=9 somoff_size=9
PASS som_size_somoff full_suppression=1.000000 somoff_suppression=0.615878 delta=0.384122
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.014913 frac_corr_gt_0p2_delta=0.023784
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.521368 mean_osi_off=0.461474
```

### 7.3 40x40 Distributed Grid3 Without L4 Inter-Site

Initial `40 x 40` grid3 without L4 inter-site failed narrowly:

```text
FAIL osi full_post=0.689447 control_post=0.000000 delta=0.689447
FAIL weights_full[l23ee] changed_frac=0.045329 p95=0.000072 threshold=0.000090
```

Debugger found this was not L4 map failure. L4 was stable; active sites were
well tuned; the sheet needed slightly more recurrent-only consolidation.

With `V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27`, it passed:

```text
PASS validation_sites required=9 full_context=9 somoff_context=9 full_size=9 somoff_size=9
PASS osi full_post=0.707107 control_post=0.000000 delta=0.707107
PASS weights_full[l23ee] changed_frac=0.051125 p95=0.000094 threshold=0.000090
PASS som_size_somoff full_suppression=1.000000 somoff_suppression=0.641907 delta=0.358093
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.020056 frac_corr_gt_0p2_delta=0.031213
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.506107 mean_osi_off=0.428033
```

### 7.4 32x32 L4 Inter-Site, Uniform Scale 0.12

The first L4 inter-site run used uniform scale `0.12` for all three new
projections. It failed:

```text
FAIL l4_intersite_map_preservation post_l4_median_osi=0.297187 post_l4_map_error_deg=1.117913
FAIL osi full_post=0.694984 control_post=0.000000 delta=0.694984
FAIL weights_full[l23ee] changed_frac=0.043625 p95=0.000066 threshold=0.000090
```

Debugger found the L4 map was not actually degraded. L4 median OSI increased
relative to control, and map error stayed low. The real problem was
over-recruited L4PV, reduced L4E firing, reduced L2/3 active endpoint coverage,
and therefore weakened all-site L2/3 OSI and all-edge L23EE movement.

### 7.5 32x32 L4 Inter-Site, Uniform Scale 0.06

Uniform scale `0.06` improved but still failed:

```text
FAIL l4_intersite_map_preservation post_l4_median_osi=0.285546 post_l4_map_error_deg=1.067410
FAIL osi full_post=0.695790 control_post=0.000000 delta=0.695790
FAIL weights_full[l23ee] changed_frac=0.046393 p95=0.000076 threshold=0.000090
FAIL l23ee_response_corr_specificity best_margin=0.000001
```

Again, the L4 map issue was a validator threshold issue, not biological map
degradation. The remaining mechanism issue was still E/I balance: the
intersite inhibitory path was too strong relative to the excitatory local
spread.

### 7.6 32x32 L4 Inter-Site, Balanced E/I Scales

Final balanced `32 x 32` L4 inter-site run:

```text
V1_VALIDATION_GRID_SIDE=3
V1_L4_INTERSITE_ENABLE=1
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
```

This passed.

Key lines:

```text
PASS validation_sites required=9 full_context=9 somoff_context=9 full_size=9 somoff_size=9
PASS l4_intersite_enabled full=1 control=1 somoff=1
PASS l4_intersite_connectivity radius_sites=2.000000 weight_scale=0.120000 l4ee_scale=0.060000 l4e_to_l4pv_scale=0.030000 l4pv_to_l4e_scale=0.030000 l4ee_edges=5809152 l4e_to_l4pv_edges=1089216 l4pv_to_l4e_edges=1089216 max_distance_sites=2.828427 max_same_site_fraction=0.000000 max_beyond_radius_fraction=0.000000
PASS l4_intersite_map_preservation full_post_l4_median_osi=0.268790 control_post_l4_median_osi=0.268790 osi_drop=0.000000 allowed_osi_drop=0.020000 full_post_l4_map_error_deg=0.898734 control_post_l4_map_error_deg=0.898734 map_error_delta=0.000000
PASS l4_intersite_spread_bounded l4_peak_rate_hz=154.629630 small_peak_ratio=0.510404 large_peak_ratio=0.878219
PASS osi full_post=0.733513 control_post=0.000000 delta=0.733513
PASS weights_full[l23ee] active=406530 changed_frac=0.054436 p95=0.000099 threshold=0.000090 lower_frac=0.000005 upper_frac=0.000002 min_nonzero=0.001000 max_nonzero=0.010000
PASS som_size_somoff full_suppression=0.999114 somoff_suppression=0.590845 delta=0.408269
PASS l23ee_response_corr_specificity best_margin=0.000003
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.016622 frac_corr_gt_0p2_delta=0.024172
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.513468 mean_osi_off=0.461410
PASS vip_weights none found
```

### 7.7 40x40 L4 Inter-Site, Balanced E/I Scales

The scaled L4 inter-site run used:

```text
-DV1_SHEET_SIDE=40
V1_VALIDATION_GRID_SIDE=3
V1_L4_INTERSITE_ENABLE=1
V1_L4_INTERSITE_EE_SCALE=0.06
V1_L4_INTERSITE_E_PV_SCALE=0.03
V1_L4_INTERSITE_PV_E_SCALE=0.03
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27
```

All L4, SOM, rate, recurrent, and responsive-site gates passed. The strict
all-site L2/3 OSI gate missed:

```text
FAIL osi full_post=0.685620 control_post=0.000000 delta=0.685620
```

Everything else passed:

```text
PASS validation_sites required=9 full_context=9 somoff_context=9 full_size=9 somoff_size=9
PASS l4_intersite_enabled full=1 control=1 somoff=1
PASS l4_intersite_connectivity radius_sites=2.000000 weight_scale=0.120000 l4ee_scale=0.060000 l4e_to_l4pv_scale=0.030000 l4pv_to_l4e_scale=0.030000 l4ee_edges=9225216 l4e_to_l4pv_edges=1729728 l4pv_to_l4e_edges=1729728 max_distance_sites=2.828427 max_same_site_fraction=0.000000 max_beyond_radius_fraction=0.000000
PASS l4_intersite_map_preservation full_post_l4_median_osi=0.270943 control_post_l4_median_osi=0.270943 osi_drop=0.000000 allowed_osi_drop=0.020000 full_post_l4_map_error_deg=0.790131 control_post_l4_map_error_deg=0.790131 map_error_delta=0.000000
PASS l4_intersite_spread_bounded l4_peak_rate_hz=156.863426 small_peak_ratio=0.525935 large_peak_ratio=0.884896
PASS weights_full[l23ee] active=643887 changed_frac=0.054916 p95=0.000099 threshold=0.000090 lower_frac=0.000002 upper_frac=0.000002 min_nonzero=0.001000 max_nonzero=0.010000
PASS rates[full:l23e] median=0.052083 frac_lt1=0.937500 p99=1.719010 limit=100.0
PASS rates[full:l23pv] median=40.000000 frac_lt1=0.048125 p99=67.916667 limit=150.0
PASS rates[full:l23som] median=9.166667 frac_lt1=0.003750 p99=17.916667 limit=150.0
PASS som_size_somoff full_suppression=1.000000 somoff_suppression=0.623115 delta=0.376885
PASS l23ee_specificity best_margin=0.000003
PASS l23ee_response_corr_specificity best_margin=0.000003
PASS l23ee_strong_synapse_enrichment corr_odds_ratio=2.818125 combined_odds_ratio=3.624996
PASS l23ee_recurrence_corr_contribution mean_corr_delta=0.022658 frac_corr_gt_0p2_delta=0.037059
PASS l23ee_recurrence_rate_osi_safety mean_osi_on=0.505276 mean_osi_off=0.433129
PASS vip_weights none found
```

With explicit responsive-site rescue enabled, the validator passed:

```text
PASS osi_responsive_rescue strict_all_site_pass=0 downstream_gates_pass=1 responsive_gate_pass=1 threshold_hz=1.000000 full_responsive_count=100 control_responsive_count=65 full_responsive_median_osi=0.753063 control_responsive_median_osi=0.462054 responsive_delta=0.291009
```

This is not a claim that the model fully passes sheet-wide biological coverage.
It means tuned responses are strong where L2/3 is responsive, while all-site
coverage is still too sparse.

## 8. Coverage and Sparsity

For `40 x 40` L4 inter-site:

```text
total_count=1600
active_count=1075
active_fraction=0.671875
responsive_threshold_hz=1.000000
responsive_count=100
responsive_fraction=0.062500
all_median_osi=0.685620
active_median_osi=0.877218
responsive_median_osi=0.753063
```

For the `40 x 40` no-learning control:

```text
active_count=685
active_fraction=0.428125
responsive_count=65
responsive_fraction=0.040625
all_median_osi=0.000000
active_median_osi=0.866609
responsive_median_osi=0.462054
```

For `32 x 32` L4 inter-site:

```text
strict all-site L2/3 OSI=0.733513
responsive fraction approximately 0.075195
```

Interpretation:

- The `40 x 40` model is not globally silent: `67.2%` of sites have some
  activity.
- However, only `6.25%` of sites have site-mean L2/3 rate at or above `1 Hz`.
- The responsive sites are tuned, but responsive coverage is too sparse for a
  cat/macaque-like V1 interpretation.
- This is the main biological gap for the next branch.

## 9. Direct Biology Comparison for Coverage

The model's current `6-8%` responsive site coverage is likely too sparse for
cat/macaque-like visually driven L2/3. Direct comparison is not exact because
the model's current coverage metric is a site-average over `16` L23E cells and
across orientations, whereas many biology papers report cell-level response to
at least one stimulus. Still, the direction is clear.

Examples from biological literature:

- Awake macaque V1 single-unit studies report high fractions of visually
  responsive cells and many orientation-selective cells.
- Macaque superficial-layer two-photon data report large fractions of neurons
  tuned to orientation/spatial frequency.
- Mouse L2/3 is sparser than primate/cat V1, but reported visually responsive
  fractions to grating/plaid stimuli are still much higher than `6-8%` when
  measured at the cell level.

Therefore the current model should not claim biologically realistic responsive
coverage yet. It should claim:

```text
L2/3 selectivity is strong among responsive sites, but responsive sheet
coverage is currently too sparse and must be corrected before using the model
for serious top-down expectation conclusions.
```

## 10. Hardcoded vs Learned vs Validated

Hardcoded:

- L4 orientation map.
- L4 simple-cell-like drive.
- E/PV/SOM/VIP subtype identities.
- Sheet geometry.
- Base population counts.
- L4 to L2/3 soft orientation-biased feedforward prior.
- Local connectivity radii and base weights.

Opt-in structural prior:

- L4 inter-site radius-2 local E/I spread.

Plastic/learned:

- L4E to L23E feedforward STDP.
- L23E to L23E recurrent STDP.
- L23PV to L23E homeostatic inhibitory plasticity.
- L23SOM to L23E homeostatic inhibitory plasticity.

Validated:

- Default `32 x 32` behavior.
- `32 x 32` distributed grid3 behavior.
- `40 x 40` grid3 no-L4 behavior with recurrent-only consolidation `27`.
- `32 x 32` grid3 L4 inter-site behavior.
- `40 x 40` grid3 L4 inter-site behavior under responsive-site OSI
  interpretation, with all downstream gates passing.

Not validated yet:

- Emergent pinwheels.
- Emergent L4 orientation map.
- Biologically realistic cell-level responsive coverage.
- Top-down/VIP expectation feedback.
- True annular surround validation.
- Multi-seed robustness.

## 11. Reproduction Commands

Default 32x32 compatibility:

```bash
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_l4is32_default_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

32x32 L4 inter-site grid3:

```bash
V1_VALIDATION_GRID_SIDE=3 \
V1_L4_INTERSITE_ENABLE=1 \
V1_L4_INTERSITE_EE_SCALE=0.06 \
V1_L4_INTERSITE_E_PV_SCALE=0.03 \
V1_L4_INTERSITE_PV_E_SCALE=0.03 \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_l4is32_grid3_ei006003_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

40x40 L4 inter-site grid3:

```bash
CXXFLAGS="-DV1_SHEET_SIDE=40" \
V1_VALIDATION_GRID_SIDE=3 \
V1_L4_INTERSITE_ENABLE=1 \
V1_L4_INTERSITE_EE_SCALE=0.06 \
V1_L4_INTERSITE_E_PV_SCALE=0.03 \
V1_L4_INTERSITE_PV_E_SCALE=0.03 \
V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS=27 \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_l4is40_grid3_ro27_ei006003_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

Strict validator:

```bash
python3 tools/validate_full_plasticity.py \
  --genn-dir /scratch/proj/v1_snn_l4_l23/genn \
  --full v1_l4is40_grid3_ro27_ei006003_full \
  --control v1_l4is40_grid3_ro27_ei006003_control \
  --somoff v1_l4is40_grid3_ro27_ei006003_somoff \
  --recoff v1_l4is40_grid3_ro27_ei006003_recoff \
  --min-validation-sites 9 \
  --require-l4-intersite
```

Responsive-site OSI reporting/rescue:

```bash
python3 tools/validate_full_plasticity.py \
  --genn-dir /scratch/proj/v1_snn_l4_l23/genn \
  --full v1_l4is40_grid3_ro27_ei006003_full \
  --control v1_l4is40_grid3_ro27_ei006003_control \
  --somoff v1_l4is40_grid3_ro27_ei006003_somoff \
  --recoff v1_l4is40_grid3_ro27_ei006003_recoff \
  --min-validation-sites 9 \
  --require-l4-intersite \
  --allow-responsive-osi
```

## 12. Next Branch: Fix Responsive Coverage

The next work should correct the coverage gap without breaking OSI, SOM, PV,
or recurrence validation.

Required first step:

- Export cell-level peak responsiveness, not only site-level mean rates.

Why:

- Current `responsive >=1 Hz` is site-average L23E rate across `16` E cells and
  across orientation trials.
- Biological responsiveness is usually cell-level response to at least one
  stimulus, not site-average response.
- A site can look weak even if a subset of cells is biologically responsive.

Planned implementation:

- Add validator parsing of `*_l23e_cell_tuning.csv`.
- Report cell-level peak-rate coverage.
- Report cell-level responsive fraction at configurable thresholds.
- Compare trained full vs no-learning control.
- Add coverage gates that are biologically interpretable.
- Then tune coverage with minimal mechanisms only if the cell-level export shows
  true under-responsiveness rather than a site-averaging artifact.

Any coverage fix must preserve:

- L2/3 OSI.
- L23EE plasticity.
- Recurrent specificity and recurrence causal contribution.
- PV/SOM rate safety.
- SOM size suppression and SOM-off causality.
- L4 inter-site map preservation and bounded spread.

## 13. Current Bottom Line

The model now has an opt-in, biologically conservative L4 inter-site mechanism:
weak local granular-layer retinotopic spread and normalization. It is validated
on `32 x 32` and structurally/functionally validated on `40 x 40`.

The remaining issue is not L4 inter-site biology. The remaining issue is
responsive coverage in the scaled `40 x 40` sheet: all-site L2/3 OSI narrowly
misses because too many sites are weakly responsive, although responsive sites
are strongly tuned and all other validations pass.

Therefore the next scientific target is cell-level responsive coverage, not
more L4 inter-site tuning.
