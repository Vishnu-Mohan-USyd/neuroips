# L2/3 population-state validation on dev2

Date: 2026-06-10 Australia/Sydney

Branch: `v1-l23-population-state-validation`

This report records the first strict validation of stable L2/3 population-state
coding as the primary target for future higher-area prediction. The validation
does not use exact top-k tile repeatability as the main biological pass/fail
criterion. Exact top-k remains a diagnostic stress test only.

## Rationale

Biological V1/L2/3 natural-stimulus responses are sparse and trial-variable at
single-cell level, while downstream-readable population geometry can remain
stable. The validation target is therefore a distributed population state:

- matched-repeat L2/3E population-vector reliability,
- odd/even repeat representational similarity matrix stability,
- held-out-repeat decoder/readout transfer,
- deterministic temporal and spatial shuffle controls,
- sparse but distributed activity,
- no future-frame, target-label, validation-metric, or HVA feedback leakage.

This follows the biological framing in Kampa et al. 2011, Yoshida and Ohki
2020, Xia et al. 2021, Froudarakis et al. 2014, Churchland et al. 2010, and
standard RSA practice from Kriegeskorte et al. 2008.

## Implementation

Added:

```text
tools/validate_l23_population_state.py
docs/l23_population_state_validation.md
tests/test_l23_population_state_validator.py
```

The validator consumes existing GeNN `*_video_site_rates.csv` exports and an
optional matching `*_video_frame_summary.csv`. It does not change the model. It
can crop a central validation core with:

```bash
--sheet-side 40 --core-side 32
```

Metrics are emitted with optional user-supplied thresholds. Threshold failures
return exit code `1`; schema/input failures return exit code `2`.

## Local tool verification

Commands run locally:

```bash
python -m py_compile tools/validate_l23_population_state.py
python -m unittest tests.test_l23_population_state_validator
PYTHONPATH=src python -m unittest discover -s tests
git diff --check -- tools/validate_l23_population_state.py docs/l23_population_state_validation.md tests/test_l23_population_state_validator.py
```

Results:

```text
Ran 4 tests in 0.031s OK
Ran 8 tests in 0.671s OK
git diff --check: pass
```

## dev2 run

Workspace: `dev2`

Run:

```text
v1_l23_popstate4_dev2_20260609T154837Z
```

Artifacts:

```text
Run dir:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z

Run log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z.log

Validator log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_validator.log

Validator JSON:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_metrics.json
```

The GeNN run completed cleanly. The log ends with:

```text
model build complete
```

## Run configuration sanity

Selected summary rows:

```text
validation_sheet_side,40
validation_core_enabled,1.000000
validation_core_side,32
validation_core_offset_x_sites,4
validation_core_offset_y_sites,4
final_post_video_l23_median_osi,0.790750
video_replay_enabled,1.000000
video_frame_count,64
video_repeat_count,4
video_l4_drive_scale,0.850000
video_feedback_disabled,1.000000
video_training_enabled,1.000000
video_ff_event_trace_enabled,1.000000
video_ff_event_trace_future_frame_used,0.000000
video_ff_event_trace_target_label_used,0.000000
video_ff_event_trace_heldout_frames_used,0.000000
video_ff_event_trace_hva_feedback_enabled,0.000000
video_l23ee_triplet_homeostatic_plasticity_enabled,1.000000
video_l23ee_triplet_homeostatic_plasticity_future_frame_used,0.000000
video_l23ee_triplet_homeostatic_plasticity_target_label_used,0.000000
video_l23ee_triplet_homeostatic_plasticity_heldout_frames_used,0.000000
video_l23ee_triplet_homeostatic_plasticity_validation_metric_used,0.000000
video_pv_reliability_output_scale,1.050000
video_som_reliability_output_scale,0.900000
lower_v1_video_consolidation_enabled,1.000000
lower_v1_video_consolidation_heldout_fraction,0.156250
lower_v1_video_consolidation_hva_feedback_enabled,0.000000
```

Interpretation:

- The validation used a 40x40 sheet with the central 32x32 core.
- HVA/top-down feedback was disabled.
- Future frames, target labels, held-out frames, and validation metrics were
  not used by the recurrent triplet rule or event-driven feedforward path.
- Final post-video L2/3 OSI remained high at `0.790750`.

## Strict population-state gates

Validator command used:

```bash
python tools/validate_l23_population_state.py \
  /scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z/v1_l23_popstate4_dev2_20260609T154837Z_video_site_rates.csv \
  --population l23e \
  --sheet-side 40 \
  --core-side 32 \
  --seed 17 \
  --shuffle-count 100 \
  --top-k 5 \
  --threshold repeat_count>=4 \
  --threshold frame_count>=64 \
  --threshold repeat_vector_corr_mean>=0.45 \
  --threshold repeat_frame_shuffle_gap_mean>=0.20 \
  --threshold repeat_site_shuffle_gap_mean>=0.15 \
  --threshold odd_even_rsm_corr>=0.50 \
  --threshold odd_even_rsm_frame_shuffle_gap_mean>=0.15 \
  --threshold odd_even_rsm_site_shuffle_gap_mean>=0.15 \
  --threshold heldout_decoder_top1_accuracy>=0.20 \
  --threshold heldout_decoder_top5_accuracy>=0.55 \
  --threshold heldout_decoder_top1_shuffle_gap_mean>=0.10 \
  --threshold heldout_decoder_top5_shuffle_gap_mean>=0.15
```

Result: `VALIDATOR_EXIT=0`.

All thresholds passed:

| Metric | Value | Gate |
| --- | ---: | ---: |
| `repeat_count` | `4` | `>=4` |
| `frame_count` | `64` | `>=64` |
| `site_count` | `1024` | central core |
| `repeat_vector_corr_mean` | `0.672147` | `>=0.45` |
| `repeat_frame_shuffle_gap_mean` | `0.389603` | `>=0.20` |
| `repeat_site_shuffle_gap_mean` | `0.672036` | `>=0.15` |
| `odd_even_rsm_corr` | `0.901051` | `>=0.50` |
| `odd_even_rsm_frame_shuffle_gap_mean` | `0.895266` | `>=0.15` |
| `odd_even_rsm_site_shuffle_gap_mean` | `0.897045` | `>=0.15` |
| `heldout_decoder_top1_accuracy` | `0.750000` | `>=0.20` |
| `heldout_decoder_top5_accuracy` | `0.996094` | `>=0.55` |
| `heldout_decoder_top1_shuffle_gap_mean` | `0.734023` | `>=0.10` |
| `heldout_decoder_top5_shuffle_gap_mean` | `0.916406` | `>=0.15` |

Additional diagnostics:

```text
heldout_decoder_chance_top1=0.015625
heldout_decoder_chance_top5=0.078125
heldout_decoder_decoded_count=256
fano_count_source=rate_hz_x_frame_duration
fano_mean=0.108380
fano_early_minus_late=-0.003603
```

The Fano early/late value is reported as a diagnostic only. This artifact does
not contain an explicit blank-to-stimulus transition design for a strict
Churchland-style variability-quenching gate.

## Conclusion

The lower V1 L2/3 representation passes the biology-aligned population-state
validation on `dev2`: stable repeated population vectors, stable odd/even
representational geometry, and high held-out-repeat readout transfer, all far
above deterministic temporal and spatial shuffle controls. This supports using
the distributed L2/3 population state as the higher-area prediction target.

This does not claim that exact top-k tile identity is the correct biological
target. It also does not yet validate higher-area prediction or feedback; it
only validates the lower-V1 repeated natural-video population-state target.
