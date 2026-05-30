# HVA Predictor Scientific Audit

Validated prefix:

```text
v1_hva_pred18_l23_topk_weighted_delay8_som051_256f
```

Validator status:

```text
full:0
control:0
somoff:0
recoff:0
pvoff:0
validator:0
```

This audit is intentionally narrow. It assesses whether the current
predictor-only HVA sidecar makes a non-cheating future L23E population-state
prediction while leaving lower V1 unchanged. It does not claim biologically
complete HVA circuitry or top-down expectation.

## No-Cheat And Isolation Evidence

The HVA sidecar is host-side and predictor-only. It reads causal lower-V1 L23E
tile history after natural-video replay and never writes back to V1.

Validator isolation evidence:

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

Shortcut exclusions enforced by validator and artifact schema:

```text
topk_input_channel_l23e_only=1
input_channel_l4e_enabled=0
input_channel_l23pv_enabled=0
feature_uses_non_l23_inputs=0
feature_future_leakage_enabled=0
lag_feature_future_lookahead_frames=0
topk_target_channel_l23e_only=1
l4e_target_channel_enabled=0
l23pv_target_channel_enabled=0
l23som_target_channel_enabled=0
topk_feedback_enabled=0
```

The held-out split is train-first/evaluate-second:

```text
heldout_mode_code=2
train_then_heldout_enabled=1
evaluation_updates_enabled=0
heldout_start_frame=192
train_rows=53100
heldout_rows=16800
boundary_gap_prediction_count=4500
```

The boundary-gap rows are quarantined/skipped rows, not leakage. For `pred18`,
the final split has `3` replay repeats, `100` tiles, and `15` boundary frames
per repeat, giving `3 * 100 * 15 = 4500` skipped rows. These rows are before
the held-out input block but have delayed/future labels that would touch the
held-out block, so they are excluded from both updates and held-out scoring.

The horizon audit uses the maximum future target horizon:

```text
PASS hva_predictor_topk_horizon_safety
future_target_horizon_frames=8
expected_future_target_horizon_frames=8
topk_split_safety_horizon_frames=8
event_window_frames=8
topk_future_window_frames=2
```

The top-k head is local/readout-limited, not dense all-to-all:

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

## Held-Out Top-K Prediction Evidence

The primary task is future L23E tile-pattern prediction. Inputs are current and
past L23E tile activity. The target is the top `5` future L23E tiles, delayed
by `8` video frames with a `2`-frame target window.

Strict validator result:

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

Approximate Wilson 95% confidence intervals for recall use
`168 * 5 = 840` held-out positive tile labels. These intervals are an audit
summary from reported validator counts, not a replacement for a future
bootstrap-over-samples validator.

| Metric | Hits / 840 | Recall | Approx. 95% CI |
|---|---:|---:|---:|
| Model | 226 | 0.269048 | [0.240159, 0.300039] |
| Persistence | 164 | 0.195238 | [0.169845, 0.223406] |
| Train frequency | 181 | 0.215476 | [0.189000, 0.244543] |
| No learning | 181 | 0.215476 | [0.189000, 0.244543] |
| Temporal shuffle | 107 | 0.127381 | [0.106518, 0.151636] |
| Spatial shuffle | 24 | 0.028571 | [0.019274, 0.042161] |
| Chance expectation | 42 | 0.050000 | [0.037201, 0.066896] |

Interpretation of the top-k result:

- The model passes the validator's required point-estimate margins: recall is
  `37.8049%` above persistence and `24.8619%` above train frequency.
- Recall is `5.380952x` chance and far above spatial-shuffle control.
- NDCG and MRR are also above train frequency in the final weighted-target
  run, addressing the earlier `pred16` failure mode where uniform positive
  mass passed recall/NDCG but under-ranked the strongest future tile.
- The CIs overlap between model and train frequency, so the claim should be
  treated as a strict single-run validator pass, not as a population-level
  statistical proof. A future validator should export bootstrap CIs for recall,
  NDCG, and MRR over held-out samples and repeated clips.

## Lower-V1 Preservation Evidence

The same final validator preserved the pre-top-down lower-V1 gates.

Selectivity:

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

PASS pv_gain_normalization_selectivity_safety
full_median_osi=0.400772
pvweak_median_osi=0.428447
median_osi_drop=-0.027675
median_pref_shift_deg=0.403047
max_pref_shift_deg=3.299650
```

SOM size/surround:

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

Recurrent L23E diagnostics:

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

Natural-video replay and event timing:

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

## Biology Comparison Scope

What is biologically motivated in the current sidecar:

- It predicts future lower-area activity from local, causal L23E population
  history, consistent with a higher visual area receiving feedforward cortical
  activity and forming predictive state estimates.
- It uses retinotopic tiling and local readout structure rather than dense
  arbitrary all-to-all access.
- It preserves lower-V1 physiology and validates that prediction does not come
  from hidden feedback, weight mutation, or non-L23 shortcut targets.

What is still an engineering approximation:

- The HVA predictor is host-side and rate/readout based, not a spiking HVA
  population with its own E/I microcircuit.
- The listwise top-k objective is a task-level prediction loss, not a
  demonstrated cortical learning rule.
- Signed host weights and softmax scoring are engineering abstractions.
- The target is future L23E tile activity, not direct sensory input, pixels,
  rewards, actions, or a full predictive-coding error population.

## Defensible Claims

The following claims are supported by the current evidence:

- A default-off, isolated HVA-like sidecar can learn a held-out future L23E
  tile-pattern signal from causal L23E-only history.
- The final run beats persistence, train-frequency, no-learning, temporal
  shuffle, spatial shuffle, and chance controls on the validator's top-k gates.
- The predictor does not mutate lower-V1 weights or outputs and has no V1
  feedback path.
- The lower-V1 biological validation stack remains intact in the final HVA
  predictor run.

## Claims Not Yet Supported

The current branch does not support these claims:

- HVA-to-V1 top-down expectation exists.
- Feedback improves V1 coding, suppresses expected input, or enhances mismatch.
- VIP-mediated feedback disinhibition is implemented.
- The HVA sidecar is a biologically faithful spiking HVA circuit.
- The model learns from multi-clip or multi-seed natural-video statistics.
- The reported top-k effect has been established with bootstrap CIs across
  independent clips/seeds.
- The direct L4E drive captures retina/LGN transformations or absolute visual
  latency.

## Required Next Validations Before Feedback Claims

Before making feedback or expectation claims, the next branch should add and
validate at least:

- Predictor-off versus predictor-on lower-V1 invariance repeated at the same
  prefix level after any feedback path is introduced.
- HVA-to-V1 feedback ablation, with feedback weights/currents explicitly
  exported and verified.
- Matched expected, unexpected, and shuffled-context video assays.
- VIP/PV/SOM-specific feedback perturbation controls if inhibitory feedback
  motifs are added.
- Multi-clip or multi-seed held-out evaluation, or an explicit deterministic
  seed/salt mechanism with repeated runs.
- Bootstrap CIs over held-out samples/clips for recall, NDCG, MRR, and any
  feedback-induced V1 change.
- Tests that feedback does not hardcode the target by using future frames,
  labels, optical flow, odometry, or target-channel shortcuts.

Until those validations pass, the correct interpretation is:

```text
The current system demonstrates isolated future lower-V1 L23E population-state
prediction from causal L23E history, not top-down expectation in V1.
```
