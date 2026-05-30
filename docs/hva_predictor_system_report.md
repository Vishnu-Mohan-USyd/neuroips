# HVA Predictor System Report

Branch snapshot: `v1-hva-predictor-pred18-report`

Primary validated artifact directory:

```text
/scratch/proj/v1_snn_l4_l23/genn
```

Final passing run:

```text
v1_hva_pred18_l23_topk_weighted_delay8_som051_256f
```

Validator result:

```text
full:0
control:0
somoff:0
recoff:0
pvoff:0
validator:0
```

Logs:

```text
/scratch/proj/v1_snn_l4_l23/genn/v1_hva_pred18_l23_topk_weighted_delay8_som051_256f_pipeline.log
/scratch/proj/v1_snn_l4_l23/genn/v1_hva_pred18_l23_topk_weighted_delay8_som051_256f_validator.log
```

This report documents the first higher-area milestone: a minimal
predictor-only HVA-like sidecar that predicts future L2/3 activity from causal
L2/3 history during natural-video replay. It deliberately does not feed back
into V1. The goal is to validate that a higher area can learn a future lower-V1
state signal before any top-down expectation or feedback pathway is added.

## Executive Summary

The final system keeps the existing lower V1 model fixed and adds a host-side,
default-off HVA predictor sidecar. The sidecar reads only binned L23E activity
from natural-video replay, aggregates it into retinotopic tiles, and learns a
local readout that predicts which L23E tiles will be most active 8 frames into
the future. It uses no L4, PV, SOM, VIP, motor, label, optical-flow, or
feedback signal as a target or input.

The final validation passed all strict gates: HVA isolation, future-horizon
safety, local readout structure, L23E top-k future prediction, lower-V1 OSI,
PV gain normalization, SOM recruitment and SOM ablation, recurrent L23E
specificity and contribution, natural-video sparsity, and millisecond-scale
laminar timing. The final predictor beat persistence, train-frequency,
no-learning, temporal shuffle, spatial shuffle, and chance controls under the
validator's top-k metrics.

## Lower-V1 Base Model

The HVA predictor is attached only after lower-V1 replay spike counts are
available. The lower model remains the reduced two-sheet V1 scaffold documented
in `docs/pre_top_down_v1_biology_alignment_report.md`:

- L4 and L2/3 are aligned retinotopic sheets.
- The validated sheet is `40 x 40 = 1600` retinotopic sites.
- Per site: `16` L4E, `3` L4PV, `1` L4SOM, `16` L23E, `2` L23PV,
  `1` L23SOM, and `1` L23VIP.
- L4 receives fixed simple-cell-like oriented input drive.
- L4 to L2/3 feedforward, L23E recurrence, PV, and SOM mechanisms are trained
  and validated before HVA prediction.
- Natural-video replay uses precomputed L4E drive generated from KITTI Raw
  frames through a fixed 4-orientation x 4-phase Gabor/simple-cell filter bank.

The HVA predictor does not alter any lower-V1 weights or replay responses.
This is enforced by validator fingerprints and mutation checks.

## Dataset And Replay Drive

The final HVA validation used a 256-frame KITTI Raw-derived L4E drive:

```text
/home/vishnu/datasets/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
```

The drive metadata:

```text
frame_count=256
sheet_side=40
k_num_l4e=25600
l4e_per_site=16
filter_bank=4 orientations x 4 phases, zero_mean_unit_norm_gabor
drive_scale=1.0
drive_offset=0.12
clip=[0,1]
```

KITTI is used only as unsupervised natural-video input. No labels, object
annotations, odometry, optical flow, motor state, or future frames are used as
HVA inputs.

## HVA Predictor Architecture

The HVA module is a predictor-only sidecar implemented in
`genn/v1TwoLayerModel.cc`. It is not a GeNN spiking population yet. It is a
host-side readout trained from replayed lower-V1 spike counts after lower-V1
activity has been collected.

Main code entry points:

```text
HVAPredictorConfig
getHVAPredictorConfig(...)
trainHVAPredictorSidecar(...)
HVAPredictorRateRow
HVAPredictorPredictionRow
HVAPredictorEventTileRow
HVAPredictorResult
```

The final predictor configuration:

```text
V1_HVA_PREDICTOR_ENABLE=1
V1_HVA_PREDICTOR_TILE_SIZE_SITES=4
V1_HVA_PREDICTOR_DELAY_FRAMES=8
V1_HVA_TOPK_FUTURE_WINDOW_FRAMES=2
V1_HVA_TOPK_K=5
V1_HVA_TOPK_LEARNING_RATE=0.005
V1_HVA_TOPK_WEIGHT_DECAY=0.001
V1_HVA_PREDICTOR_LOCAL_RADIUS_TILES=1
V1_HVA_PREDICTOR_EPOCHS=5
V1_HVA_FEATURE_LAG_COUNT=5
V1_HVA_FEATURE_CONTEXT_RADIUS_TILES=1
V1_FEEDBACK_ENABLE=0
V1_HVA_FEEDBACK_ENABLE=0
```

Tile geometry:

```text
V1 sheet: 40 x 40 sites
HVA tile size: 4 x 4 V1 sites
HVA tile grid: 10 x 10
HVA tile count: 100
Top-k target count: 5 tiles
```

The sidecar uses a local readout. Each target tile reads only from nearby
source tiles within `local_radius_tiles=1`, producing `460` active tile-to-tile
readout pairs and `0` distant nonzero pairs in the final run.

## HVA Inputs

Inputs are L23E-only. For each tile, the sidecar computes causal features from
past and current L23E tile activity:

- Current L23E tile rate.
- Fast, medium, and slow causal traces.
- Causal derivative.
- Causal lag-history features.
- Local-context summaries over nearby L23E tiles.

The validated trace constants are:

```text
fast_trace_tau_ms=50
medium_trace_tau_ms=150
slow_trace_tau_ms=500
```

The validator enforces:

```text
topk_input_channel_l23e_only=1
input_channel_l4e_enabled=0
input_channel_l23pv_enabled=0
feature_uses_non_l23_inputs=0
feature_future_leakage_enabled=0
lag_feature_future_lookahead_frames=0
```

## HVA Target

The primary validated target is future L23E population activity pattern, not
raw pixel prediction and not L4/PV/SOM activity prediction.

For each valid sample:

```text
input time: current and past L23E tile history
target delay: 8 video frames
target window: 2 frames
target signal: average future L23E tile activity over the target window
label: top 5 future active L23E tiles
```

At `V1_VIDEO_FRAME_MS=100`, this is an approximately `800-1000 ms` future
L23E tile-pattern prediction target.

The exact top-5 gate measures whether the model ranks the same discrete future
tiles as the held-out target. The population-distribution gate is a separate
validator-only assay: it repeat-averages held-out target frames, applies a
fixed radius-1 retinotopic smoothing kernel to the target mass on the 10x10
tile grid, and scores each unchanged model/baseline held-out sample ranking
with weighted NDCG and captured ideal mass. This asks whether the predictor captures the
future L23E population activity distribution, not whether target smoothing or
future information is used by the model.

The final model-side fix weights the listwise top-k training target by the
future L23E tile activity strength among the selected top-k tiles instead of
assigning uniform mass to all positives. This preserves the L23E-only target
while aligning the learning objective with rank-sensitive validation metrics
such as MRR and NDCG.

The validator enforces:

```text
topk_target_channel_l23e_only=1
l4e_target_channel_enabled=0
l23pv_target_channel_enabled=0
l23som_target_channel_enabled=0
topk_feedback_enabled=0
```

## Training Rule

The HVA predictor uses a simple local listwise readout update:

```text
scores = local_linear_readout(causal_L23E_features)
probabilities = softmax(scores over target tiles)
target_distribution = normalized future L23E strength over future top-k tiles
error = target_distribution - probabilities
```

Weights and biases are updated only during the training split:

```text
topk_bias += learning_rate * error
topk_weight = weight * (1 - weight_decay) + learning_rate * error * feature
```

The update is local to the readout pair mask. There is no global post-hoc
normalization, no backpropagation through V1, no surrogate-gradient SNN
training, and no feedback current.

The final predictor is deliberately minimal. It is best interpreted as the
first predictive readout of future lower-V1 population state, not as a complete
spiking higher visual cortical area.

## Train And Held-Out Split

The HVA training uses train-first/evaluate-second splitting. Held-out rows are
never used for updates.

Validation checks:

```text
heldout_mode_code=2
train_then_heldout_enabled=1
evaluation_updates_enabled=0
boundary_gap_prediction_count=4500
```

The top-k horizon safety gate uses the maximum future target horizon across
event and top-k objectives:

```text
future_target_horizon_frames=8
topk_split_safety_horizon_frames=8
event_window_frames=8
topk_future_window_frames=2
heldout_start_frame=192
train_rows=53100
heldout_rows=16800
boundary_gap_prediction_count=4500
```

This prevents training targets from crossing into held-out frames even when
the future top-k window differs from other HVA reporting windows.

The `4500` boundary-gap rows are not leakage and were not used for updates or
held-out scoring. They are quarantined rows whose input frame is before the
held-out block but whose delayed/future target window would reach the held-out
block. With `3` replay repeats, `100` HVA tiles, and `15` boundary frames per
repeat, the skipped count is `3 * 100 * 15 = 4500`. This is the intended
safety buffer between train and held-out content for the final `pred18` run.

## Final Run Configuration

The final passing run used:

```text
prefix=v1_hva_pred18_l23_topk_weighted_delay8_som051_256f
CXXFLAGS=-DV1_SHEET_SIDE=40
V1_VIDEO_REPLAY_ENABLE=1
V1_VIDEO_DRIVE_BIN=/scratch/proj/v1_snn_l4_l23/data/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
V1_VIDEO_FRAME_COUNT=256
V1_VIDEO_MAX_FRAMES=256
V1_VIDEO_FRAME_MS=100
V1_HVA_PREDICTOR_ENABLE=1
V1_HVA_PREDICTOR_DELAY_FRAMES=8
V1_HVA_TOPK_FUTURE_WINDOW_FRAMES=2
V1_L23E_SOM_BROAD_RECRUIT_ENABLE=1
V1_L23E_SOM_BROAD_RECRUIT_WEIGHT_SCALE=0.051
V1_FEEDBACK_ENABLE=0
V1_HVA_FEEDBACK_ENABLE=0
```

The SOM broad recruitment scale was set to `0.051` because `0.050` passed PV
but narrowly missed the broad-SOM recruitment threshold, while `0.055` passed
SOM but narrowly reduced the PV ablation gain below threshold. The chosen value
is an existing model parameter, not a new pathway.

## Final Validation Results

The final validator returned `rc=0` with no `FAIL` lines.

HVA isolation:

```text
PASS hva_predictor_isolation
lower_v1_frozen=1
hva_to_v1_connection_count=0
hva_to_v1_current_enabled=0
lower_v1_weight_delta_max_after_hva=0
lower_v1_output_delta_max_after_hva=0
v1_mutation_after_hva_enabled=0
fingerprint_equal=1
video_feedback_disabled=1
```

Top-k horizon safety:

```text
PASS hva_predictor_topk_horizon_safety
future_target_horizon_frames=8
expected_future_target_horizon_frames=8
topk_split_safety_horizon_frames=8
heldout_start_frame=192
```

Top-k locality:

```text
PASS hva_predictor_topk_head_locality_structure
topk_local_readout_enabled=1
topk_dense_all_to_all_readout_enabled=0
topk_local_radius_tiles=1
topk_local_pair_count=460
topk_distant_pair_count=9540
topk_local_nonzero_pair_count=460
topk_distant_nonzero_pair_count=0
topk_distant_abs_weight_sum=0
```

HVA future L23E prediction:

```text
PASS hva_predictor_l23e_future_topk_success
heldout_valid_sample_count=168
topk_k=5
tile_count=100
model_recall_at_k=0.269048
persistence_recall_at_k=0.195238
train_frequency_recall_at_k=0.215476
no_learning_recall_at_k=0.215476
time_shuffle_recall_at_k=0.127381
spatial_shuffle_recall_at_k=0.028571
chance_recall_at_k=0.050000
model_chance_ratio=5.380952
model_ndcg_at_k=0.282751
train_frequency_ndcg_at_k=0.226049
model_mrr=0.521131
train_frequency_mrr=0.449405
relative_vs_persistence=0.378049
relative_vs_train_frequency=0.248619
```

Lower-V1 OSI:

```text
PASS osi
full_post=0.786952
control_post=0.000000
delta=0.786952
```

PV gain normalization:

```text
PASS pv_gain_normalization_causality
full_mean_l23e_hz=29.328125
pvweak_mean_l23e_hz=32.882812
mean_increase_fraction=0.121204
required_gain_floor=0.100000
pvweak_l23e_context_p99_hz=52.696875
p99_limit_hz=100.000000
```

PV selectivity safety:

```text
PASS pv_gain_normalization_selectivity_safety
full_median_osi=0.400772
pvweak_median_osi=0.428447
median_osi_drop=-0.027675
median_pref_shift_deg=0.403047
max_pref_shift_deg=3.299650
```

SOM recruitment and SOM-off causality:

```text
PASS som_size_som_recruitment
peak_som_rate=32.407407
large_som_rate=14.074074
center_context_som_rate=32.777778
broad_context_som_rate=27.222222
som_recruitment_index=-0.169492

PASS som_size_somoff
full_suppression=1.000000
somoff_suppression=0.674693
delta=0.325307

PASS som_size_somoff_site_rescue
site_rescue_fraction=1.000000
full_large_l23e_rate=0.000000
somoff_large_l23e_rate=31.273148
```

L23E recurrence:

```text
PASS l23ee_response_corr_specificity
row_count=36693
active_margin_ok=1

PASS l23ee_recurrent_heavy_tail
active_count=643887
p50=0.004209
p90=0.006771
p95=0.007749
p99=0.009900
description=bounded_heavy_tailed_like

PASS l23ee_recurrent_shuffle_specificity
observed_delta=0.049709
shuffle_q95_delta=0.016385
z_score=6.363354

PASS l23ee_recurrence_corr_contribution
mean_corr_delta=0.032452
frac_corr_gt_0p2_delta=0.047881

PASS l23ee_recurrence_rate_osi_safety
peak_ratio_off_on=0.882320
mean_osi_on=0.527558
mean_osi_off=0.430384
```

Natural-video rate and timing validation:

```text
PASS natural_video_l23e_sparse_safe
l23e_mean_rate_hz=0.126822
l23e_site_p95_hz=0.625000
l23e_site_p99_hz=3.125000
l23e_site_frac_lt1=0.957114

PASS natural_video_event_l23_interneuron_latency_peak
l4_reference_onset_ms=2.000000
l23e_causal_onset_ms=28.000000
l23pv_causal_onset_ms=6.000000
l23som_causal_onset_ms=28.000000

PASS natural_video_event_rate_safety
l23e_peak_hz=0.178901
l23pv_peak_hz=3.802083
l23som_peak_hz=11.306424

PASS natural_video_event_crosscorr_null
l23e_best_corr=0.316839
l23pv_best_corr=0.509020
l23som_best_corr=0.359499
```

## Iteration History

The predictor did not pass immediately. The main lessons from the failed runs
are preserved because they affect interpretation.

Earlier attempts:

- Independent L23E rate and event-hazard prediction was dominated by sparse
  per-tile base rates and did not generalize reliably.
- Using non-L23 targets such as L4/PV/SOM was rejected as a shortcut.
- `pred13` used a 256-frame drive with event-window hazard prediction and
  failed because the model learned train structure but did not beat
  train-mean/no-learning on held-out calibration.
- `pred14` introduced top-k future L23E population prediction and passed an
  initial validation, but it did not yet include the later trust-gap audits.
- Static review then added split-horizon safety and actual top-k head locality
  validation.
- `pred15` used the trust-gap audits and failed because the top-k target was
  too near-term (`t+1..t+2`), making persistence a strong legitimate baseline.
- `pred16` set the prediction delay to 8 frames. It passed recall/NDCG against
  persistence and train-frequency but failed MRR because the top-k loss treated
  all five positive tiles uniformly.
- `pred17` weighted positive target mass by future L23E activity strength and
  passed HVA top-k, but failed PV gain after SOM broad recruitment was set too
  high at `0.055`.
- `pred18` used the weighted top-k target with SOM broad recruitment scale
  `0.051` and passed all gates.

No failure was addressed by adding feedback, using non-L23 targets, using
future inputs, or relaxing validator gates.

## Hardcoded vs Emergent Ledger

Hardcoded or manually specified:

- L4 simple-cell/Gabor drive and orientation map.
- Reduced two-sheet retinotopic geometry.
- Population identities and lower-V1 connection motifs.
- KITTI preprocessing and fixed L4E filter bank.
- HVA tile geometry, local readout radius, and top-k target definition.
- Validation gates and ablation protocols.
- SOM broad recruitment strength is a parameter selected within an existing
  mechanism to satisfy both SOM and PV gates.

Emergent or learned within this reduced system:

- Lower L23E OSI after training, under zero L4-to-L23 orientation structural
  prior in the current pre-top-down configuration.
- L23E recurrent co-tuning and correlation-enriched strong synapses.
- SOM-dependent broad/size suppression and SOM-off rescue.
- PV gain-normalization causality and selectivity safety.
- HVA top-k future L23E tile prediction from causal L23E history.

Explicitly not implemented:

- HVA spiking populations.
- HVA-to-V1 feedback currents.
- VIP-mediated feedback disinhibition.
- Prediction-error populations.
- Pixel prediction.
- Optical-flow labels, odometry, motor history, or supervised external labels.
- Dendritic compartments, apical tuft feedback, or conductance synapses.

## Cheat And Shortcut Audit

The final HVA predictor does not use the main shortcuts that were rejected
during development:

- It does not use L4, PV, SOM, or VIP activity as the prediction target.
- It does not use L4, PV, SOM, or VIP activity as input features.
- It does not receive future frames as input.
- It does not mutate lower-V1 weights or outputs.
- It does not feed any current back into V1.
- It does not use held-out rows for updates.
- It does not use dense all-to-all tile readout.
- It does not globally renormalize weights after learning.

What remains an engineering approximation:

- The HVA predictor is host-side and rate/readout based, not a spiking higher
  cortical area.
- The top-k objective is a task-level abstraction of future population state,
  not a detailed biological learning rule.
- The lower-V1 natural-video drive is a fixed filter-bank drive into L4E, not a
  full retina/LGN model.
- The HVA sidecar validates future-state prediction only. It does not yet test
  top-down expectation effects on V1.

## Code And Artifact Outputs

Main changed files:

```text
genn/v1TwoLayerModel.cc
tools/validate_full_plasticity.py
tools/datasets/prepare_video_datasets.py
docs/video_dataset_prep.md
docs/pre_top_down_v1_biology_alignment_report.md
docs/hva_predictor_system_report.md
docs/hva_predictor_scientific_audit.md
```

HVA CSV outputs:

```text
<prefix>_hva_predictor_config.csv
<prefix>_hva_predictor_metrics.csv
<prefix>_hva_predictor_rates.csv
<prefix>_hva_predictor_predictions.csv
<prefix>_hva_predictor_event_tiles.csv
<prefix>_hva_predictor_weights.csv
```

The validator consumes these files and combines them with lower-V1 summary,
rate, specificity, recurrence, SOM/PV ablation, natural-video replay, and
event-timing exports.

## Storage Cleanup

During this milestone, `/scratch` was nearly full. The dataset itself was not
the cause:

```text
/scratch/proj/v1_snn_l4_l23/data ~= 51M
```

The space was dominated by old generated diagnostic artifacts under:

```text
/scratch/proj/v1_snn_l4_l23/genn
```

Old `v1_unbiased_ff_*sampled_ff_orientation_coactivity.csv` and
`v1_unbiased_ff_*sampled_l23_training_voltage.csv` exploratory diagnostics
were deleted after the user approved cleanup. Current HVA artifacts, source,
data, and final logs were left intact. After cleanup, `/scratch` recovered to
roughly `190G` free before the final validator completed.

## Interpretation

The final claim supported by this branch is narrow:

```text
An isolated, local, L23E-only HVA-like predictor can learn to predict the
future sparse L2/3 population activity pattern during natural-video replay,
while the lower V1 circuit remains unchanged and continues to pass the
pre-top-down biological validation gates.
```

The branch does not yet support claims about top-down expectation, predictive
suppression, mismatch enhancement, VIP disinhibition, or feedback effects on
V1. Those require a later branch that adds feedback pathways and validates
matched expectation, mismatch, and feedback-ablation conditions separately.
