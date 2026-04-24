# Native CUDA Richter Dampening Report - 2026-04-24

## Scope

This report documents the current validated native C++/CUDA Richter dampening
state after the Phase 3 raw-current prediction-targeted suppression work. The
scope is deliberately limited to the native `cpp_cuda` execution path and its
post-hoc native-artifact analysis. It does not claim changes to the Brian2
model, Python training implementation, or broad assay stack outside the files
listed below.

It covers the native architecture, task/training setup,
expected/unexpected assay, `Q_active` measurement, corrected pseudo-voxel
decoder protocol, final validated `raw_ie` suppression result, speed,
validation evidence, caveats, and artifact locations.

Committed source/artifact surfaces:

- `cpp_cuda/include/expectation_snn_cuda/manifest.hpp`
- `cpp_cuda/src/richter_eval.cu`
- `cpp_cuda/src/native_cli.cpp`
- `expectation_snn/scripts/analyze_native_richter_pseudovoxels.py`
- `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_predicted_raw_suppression_sigma22_rollup_20260424.json`
- `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_pseudovoxel_transfer_decoder_robust_sigma22_20260424.json`

## Current Native CUDA Architecture

The native CUDA path implements the fast Richter assay in C++/CUDA rather than
through Python/Brian2 execution. The architecture described here is the code
that actually landed in `cpp_cuda`, not the broader intended model sketch. The
central surfaces are:

- `manifest.hpp`: native public data structures, frozen source result fields,
  and CPU/GPU/batched function signatures.
- `richter_eval.cu`: CPU reference loop, single-GPU loop, batched GPU kernels,
  seeded-source Richter execution, Q/activity telemetry, held H_pred feedback,
  feedback topology, and prediction-targeted suppression.
- `native_cli.cpp`: standalone CLI, argument parsing, richter-dampening task
  construction, artifact JSON writing, rollup-facing summary fields, and trial
  telemetry export.
- `analyze_native_richter_pseudovoxels.py`: post-hoc fMRI-like pseudo-voxel
  transfer decoder for native Richter JSON artifacts.

The production assay uses a 12-channel V1 orientation ring. Each channel has
V1_E and V1_SOM activity, plus H_context and H_prediction higher-area channels.
The landed routes are:

- sensory trailer drive to V1_E/V1_SOM through the native stimulus-rate path
- V1/H_context leader support from the frozen seeded-source schedule
- learned H_context to H_prediction checkpoint weights from the Stage 1
  checkpoint
- trailer-phase `ctx_to_pred` scatter gated off so actual trailer sensory
  drive cannot overwrite H_prediction during evaluation
- held H_prediction replay from the immediately preceding prediction window
  used as the feedback source during the trailer
- H_prediction to V1_E direct/apical feedback
- H_prediction to V1_SOM feedback with surround terms and a configurable
  same-channel center component
- optional, default-off V1_E suppression/gain surfaces used only when the
  corresponding CLI scales are nonzero

The current supported baseline keeps raw held-replay feedback active, uses
`feedback-g-total=2.0`, `feedback-r=0.3333333333333333`,
`feedback-som-center-weight=0.10`, and `v1-stim-sigma-deg=22.0`.

## Task And Training Setup

The assay reads the existing Stage 1 native checkpoint:

```text
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix2/stage1_ctx_pred_seed42_n72.json
```

Stage 0 is the preceding stabilization phase for the native checkpoint family:
it establishes the sensory/V1 substrate and inhibitory balance before sequence
learning. Stage 1 learns the context-to-prediction mapping. The leader stimulus
activates V1 and H_context; the learned H_context to H_prediction mapping
represents the forecast trailer. The Phase 3 Richter runs do not retrain either
stage. They load the frozen Stage 1 checkpoint and run a frozen evaluation.

In the Richter assay, the forecast is replayed as held H_pred feedback during
the trailer window so feedback is driven by the learned expectation rather than
by the actual trailer label. This distinction is critical: the evaluation path
is intended to test prediction-dependent dampening, not label leakage from the
held-out trailer.

The validated Phase 3 runs used:

- seed `4242`
- `--reps-expected 4`
- `--reps-unexpected 4`
- `--grating-rate-hz 100`
- `--baseline-rate-hz 0`
- `--gpu-only-production`
- raw held replay
- no energy/feedback training changes

## Expected vs Unexpected Richter Assay

The assay compares trailer responses in two conditions:

- expected: trailer matches the learned forecast for the leader
- unexpected: trailer differs from the forecast while sharing the same task
  and measurement path

The intended biological effect is prediction-related dampening: expected
trailers should show lower V1 energy/activity and weaker noisy population
readout than unexpected trailers. The validation criterion is therefore
directional and condition-resolved:

```text
expected Q_active < unexpected Q_active
expected V1 activity < unexpected V1 activity
expected noisy pseudo-voxel decoding accuracy < unexpected noisy pseudo-voxel decoding accuracy
```

The deterministic full argmax decoder is still exported for compatibility and
debugging, but it is labeled debug-only in interpretation. It operates on
high-SNR aggregate channel counts and saturates at `1.0` expected and `1.0`
unexpected in the validated condition, so it cannot distinguish dampening from
preserved linearly separable identity.

## Q_active Definition

`Q_active` is a measurement-only integrated active charge metric. Units are
`fC`, computed as `pA * ms` over the relevant phase and population.

Current V1 definitions:

- V1_E: `abs(I_e) + abs(I_i) + abs(I_ap_e)`
- V1_SOM: `abs(I_e) + abs(I_i)`
- V1 total: V1_E plus V1_SOM

The raw-current suppression patch intentionally changes the upstream raw
feedforward excitatory current before Q accumulation and before the membrane
update. This is why the final result changes real `Q_active`; it is not a
post-hoc rescaling of the metric.

## Prediction-Targeted Raw-Ie Suppression

The successful mechanism is explicit and default off:

```text
--v1-predicted-suppression-scale <float>
--v1-predicted-suppression-neighbor-weight <float>
--v1-predicted-suppression-locus effective_ie|raw_ie
```

The default behavior is preserved at scale `0.0`. The prior `effective_ie`
locus is kept intact. The validated mechanism uses:

```text
--v1-predicted-suppression-locus raw_ie
--v1-predicted-suppression-scale 16
--v1-predicted-suppression-neighbor-weight 0.5
```

During the trailer window, the prediction signal is derived only from the held
H_pred/feedback state. It does not use the actual trailer label. Suppression is
applied to the predicted channel and radius-1 neighbors, with the neighbor
contribution weighted by `0.5`. In `raw_ie` mode this gain is applied to the
raw V1_E feedforward excitatory current before `Q_active` accumulation and
before the V1_E membrane update.

Telemetry added to prove the locus and target:

- `v1_predicted_suppression_trailer_channel_signal_sum`
- `v1_predicted_suppression_trailer_channel_gain_mean`
- `v1_predicted_suppression_trailer_raw_ie_before_sum`
- `v1_predicted_suppression_trailer_raw_ie_after_sum`
- `v1_predicted_suppression_trailer_raw_ie_delta_sum`
- predicted-channel, actual-channel, and all-channel raw current delta summaries

## Pseudo-Voxel Decoder Protocol

The post-hoc utility pools 12 V1 channels into 4 contiguous pseudo-voxels, using
the same spatial binning convention as `richter_crossover._voxel_spatial_bins`.
It then trains a 6-class nearest-centroid decoder on the sensory-only
localizer trials using observed labels only: `0, 2, 4, 6, 8, 10`.

The transfer decoder tests separately on feedback-on expected and unexpected
trials. Measurement degradation is applied after voxel pooling with independent
binomial thinning at:

```text
keep_p = 0.05, 0.02, 0.015, 0.01
```

The robust analysis uses `1024` thinning seeds. Unexpected trials are partitioned
into four 24-trial subsets per seed so the unexpected accuracy is matched to
the 24 expected trials. The JSON includes:

- mean/std accuracy by condition
- unexpected-minus-expected accuracy
- true-class nearest-centroid margin
- bootstrap confidence intervals
- shuffled localizer-label chance control
- shuffled test-label chance control
- exact trial counts and quantization step

This readout is more biologically relevant than full argmax because it uses a
coarser population measurement with reduced SNR. It is not a full fMRI forward
model: it does not model hemodynamics, vascular coupling, slice acquisition, or
scanner noise. The important evidence is the condition gap under degraded
voxel-like measurements, not saturated identity recovery from high-SNR channel
counts.

The result should also not be described as representational sharpening. The
validated effect is expectation dampening: expected trials have lower
energy/activity and lower noisy pseudo-voxel decodability than unexpected
trials under the same readout. That is closer to predictive-coding response
suppression than to a claim that expected representations are sharper.

## Final Validated Result

Validated condition:

```text
rate100, sigma22, seed4242, reps_expected=4, reps_unexpected=4
feedback-g-total=2.0
feedback-r=0.3333333333333333
feedback-som-center-weight=0.10
v1-predicted-suppression-locus=raw_ie
v1-predicted-suppression-scale=16
v1-predicted-suppression-neighbor-weight=0.5
```

Native V1 energy/activity:

| Metric | Expected | Unexpected | Unexpected - Expected |
| --- | ---: | ---: | ---: |
| V1 `Q_active` fC/trial | 12,130,622.9738 | 13,409,791.7622 | 1,279,168.7884 |
| V1 activity counts/trial | 311.3333 | 449.8333 | 138.5000 |

Noisy pseudo-voxel transfer decoder:

| keep_p | Accuracy U-E | 95% bootstrap CI |
| --- | ---: | ---: |
| 0.05 | +0.065175 | [0.061777, 0.068604] |
| 0.02 | +0.104451 | [0.099477, 0.109406] |
| 0.015 | +0.111806 | [0.106557, 0.117086] |
| 0.01 | +0.111135 | [0.105357, 0.117157] |

Chance-control accuracy deltas stayed near zero in the same scale-16 analysis:

| keep_p | Shuffled localizer labels U-E | Shuffled test labels U-E |
| --- | ---: | ---: |
| 0.05 | +0.000682 | +0.000427 |
| 0.02 | +0.001119 | +0.001933 |
| 0.015 | +0.002686 | +0.001902 |
| 0.01 | -0.000549 | +0.000173 |

The full no-noise argmax decoder remains saturated:

```text
expected = 1.0
unexpected = 1.0
```

That saturation is a caveat for the old decoder, not a failure of the
pseudo-voxel criterion.

## Speed

The GPU production sweep runs were approximately `3.69 s` per rate100 reps4
condition. The selected scale-16 run recorded:

```text
run = 3.716578009 s
total_excluding_build = 3.74038373 s
write = 0.021055339 s
```

The active reps1 CPU reference parity run is much slower, as expected, but was
used only for validation.

## Validation And Parity Evidence

Build and validation were run on the remote CUDA host:

```text
/workspace/neuroips_gpu_migration_20260422/neuroips
```

Build log:

```text
/workspace/neuroips_gpu_migration_20260422/logs/native_predicted_raw_suppression_build_20260424.log
```

Parity and scale-zero equivalence log:

```text
/workspace/neuroips_gpu_migration_20260422/logs/native_predicted_raw_suppression_parity_20260424.log
```

Sweep log:

```text
/workspace/neuroips_gpu_migration_20260422/logs/native_predicted_raw_suppression_sweep_20260424.log
```

Rollup rerun log:

```text
/workspace/neuroips_gpu_migration_20260422/logs/native_predicted_raw_suppression_rollup_rerun_20260424.log
```

CPU/GPU parity passed for active `raw_ie` suppression at scale `4`,
neighbor `0.5`:

- V1 counts max diff: `0`
- suppression signal/gain max diff: `0`
- raw Ie before/after max diff: `7.45e-09`
- raw Ie delta max diff: `2.91e-11`
- Q diff: at most `7.06e-07`

Scale-zero `raw_ie` matched the supported baseline:

- V1 counts exact
- Q diffs from `0` to `7.45e-09`
- raw delta metrics zero

This validates both the active mechanism and the default-preserving behavior.

## Artifact Paths

Local committed rollups:

```text
expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_predicted_raw_suppression_sigma22_rollup_20260424.json
expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_pseudovoxel_transfer_decoder_robust_sigma22_20260424.json
```

Remote rollups:

```text
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_predicted_raw_suppression_sigma22_rollup_20260424.json
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_pseudovoxel_transfer_decoder_robust_sigma22_20260424.json
```

Selected scale-16 native artifact:

```text
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/predicted_raw_suppression_sigma22_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_feedback_r03333333333333333_gtotal20_center010_predraw_s16_n0p5.json
```

CPU/GPU parity artifacts:

```text
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/predicted_raw_suppression_sigma22_20260424/richter_dampening_fix2_n72_seed4242_reps1_rate100_sigma22_feedback_r03333333333333333_gtotal20_center010_predraw_s4_n0p5_parity_gpu_only_production.json
/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/predicted_raw_suppression_sigma22_20260424/richter_dampening_fix2_n72_seed4242_reps1_rate100_sigma22_feedback_r03333333333333333_gtotal20_center010_predraw_s4_n0p5_parity_cpu_reference.json
```

## Caveats

- The old full argmax decoder remains saturated at `1.0/1.0`; it is retained
  for continuity/debugging but should not be interpreted as fMRI-like evidence.
- The pseudo-voxel decoder is an intentionally degraded population readout, not
  a full fMRI forward model.
- The validated result is expectation dampening, not representational
  sharpening. It shows lower expected response magnitude and lower noisy
  decodability; it does not show a narrower or more precise expected
  representation.
- The validated mechanism is local and prediction-derived, but the suppression
  scale is still a native assay parameter. Further work should test robustness
  across checkpoints, sensory rates, and held-out seeds before treating the
  exact value as biologically constrained.
- The committed rollups contain summary evidence. The full per-run native JSON
  artifacts remain on the remote machine because they are larger and were not
  requested for local commit.
- The report is based on the fix2 n72 checkpoint family. It does not claim
  generalization to all Stage 1 training settings.

## Reproduction Skeleton

Traceability branch and commit:

```text
branch: cuda-richter-dampening-20260424
commit: 86d185bcd8d312a129680a8bd7db113e356304fd
```

The production validation command pattern was:

```bash
cpp_cuda/build/expectation_snn_native richter-dampening \
  --checkpoint /workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix2/stage1_ctx_pred_seed42_n72.json \
  --seed 4242 \
  --reps-expected 4 \
  --reps-unexpected 4 \
  --gpu-only-production \
  --grating-rate-hz 100 \
  --baseline-rate-hz 0 \
  --feedback-g-total 2.0 \
  --feedback-r 0.3333333333333333 \
  --feedback-som-center-weight 0.10 \
  --v1-predicted-suppression-locus raw_ie \
  --v1-predicted-suppression-scale 16 \
  --v1-predicted-suppression-neighbor-weight 0.5
```

The robust pseudo-voxel decoder is run with:

```bash
python3 expectation_snn/scripts/analyze_native_richter_pseudovoxels.py \
  --n-thinning-seeds 1024 \
  --keep-p 0.05,0.02,0.015,0.01
```

The output JSON records exact artifact inputs, trial counts, quantization step,
chance controls, and energy/activity metrics needed to reproduce the final
GO/NO-GO criterion.

## Biological Interpretation And References

The native result is interpreted as expectation dampening: a learned prediction
locally suppresses the feedforward drive for the predicted orientation
neighborhood, reducing expected-trailer V1 charge/activity and reducing
decodability only after biologically motivated measurement degradation. This is
consistent with predictive-coding accounts in which predictable sensory input
evokes smaller residual responses. The current implementation is still an
engineering assay mechanism, not a complete cortical microcircuit model.

Relevant references for interpretation:

- Rao, R. P. N. and Ballard, D. H. (1999). Predictive coding in the visual
  cortex: a functional interpretation of some extra-classical receptive-field
  effects. `Nature Neuroscience`, 2, 79-87.
- Friston, K. (2005). A theory of cortical responses. `Philosophical
  Transactions of the Royal Society B`, 360, 815-836.
- Kok, P., Jehee, J. F. M. and de Lange, F. P. (2012). Less is more:
  expectation sharpens representations in the primary visual cortex. `Neuron`,
  75, 265-270. This paper is useful context, but the current native result
  should not be claimed as sharpening.
