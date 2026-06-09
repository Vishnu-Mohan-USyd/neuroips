# L23EE Triplet/Homeostatic Plasticity Report

Branch: `v1-biophysical-l23ee-plasticity`

Accepted strict validator log:
`.runs/logs/v1_triplet_l23ee_lr125_hold15625_acceptance_strict_validator.log`

Accepted full-run artifacts:
`.runs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full/`

## Goal

The goal of this branch was to replace the previous opt-in top-k
`L23E -> L23E` heterosynaptic competition rule with a more local,
biophysically motivated recurrent plasticity mechanism. The old top-k rule
ranked incoming recurrent synapses and explicitly potentiated the top fraction.
That was useful as an engineering scaffold, but it is too algorithmic as a
biological claim: cortical synapses do not know a postsynaptic top-k list.

The replacement target was therefore a local recurrent rule that uses only
same-run L23E spiking activity during natural-video plastic exposure, updates
all existing incoming recurrent synapses continuously, and stabilizes incoming
recurrent mass with a soft postsynaptic-local homeostatic term rather than exact
global normalization.

## Implemented Mechanism

The accepted replacement is the opt-in
`V1_VIDEO_L23EE_TRIPLET_HOMEOSTATIC_PLASTICITY_ENABLE` path in
`genn/v1TwoLayerModel.cc`. It is active during recurrent-only natural-video
plastic exposure and inactive by default.

For each active `L23E -> L23E` recurrent synapse, the rule uses L23E spike
traces accumulated from the recurrent-only exposure window:

- `x_i`: presynaptic trace.
- `y_j`: postsynaptic fast trace.
- `z_j`: postsynaptic slow trace.
- LTP evidence: pre/post/slow coactivity, approximating a triplet-like term.
- LTD/timing evidence: presynaptic activity paired with postsynaptic fast
  activity.

The implemented update computes an activity score per existing recurrent edge,
then for each postsynaptic L23E cell subtracts the postsynaptic incoming mean
update. This means the competition is postsynaptic-local and continuous across
all incoming recurrent synapses, not a ranked top-k selection. A slow
postsynaptic-local mass homeostasis term nudges incoming recurrent E mass toward
that postsynaptic cell's pre-phase incoming E mass. It is not an exact
normalization step, and it does not use global network activity, validation
metrics, orientation labels, held-out frames, future frames, or HVA feedback.

Accepted rule settings in the full run:

| Setting | Value |
| --- | ---: |
| `video_l23ee_heterosyn_competition_enabled` | `0.000000` |
| `video_l23ee_heterosyn_competition_active` | `0.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_enabled` | `1.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_active` | `1.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_learning_rate` | `1.250000` |
| `video_l23ee_triplet_homeostatic_plasticity_a_plus` | `0.000001` |
| `video_l23ee_triplet_homeostatic_plasticity_a_minus` | `0.000001` |
| `video_l23ee_triplet_homeostatic_plasticity_mass_eta` | `0.050000` |
| `video_l23ee_triplet_homeostatic_plasticity_min_post_spikes` | `1.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_tau_pre_frames` | `2.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_tau_post_frames` | `2.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_tau_slow_frames` | `20.000000` |
| `video_l23ee_triplet_homeostatic_plasticity_mass_tolerance` | `0.020000` diagnostic only |
| `video_l23ee_triplet_homeostatic_plasticity_application_count` | `1` |
| `video_l23ee_triplet_homeostatic_plasticity_activity_window_count` | `216` |
| `video_l23ee_triplet_homeostatic_plasticity_active_edge_count` | `643887` |
| `video_l23ee_triplet_homeostatic_plasticity_changed_frac` | `0.014486` |
| `video_l23ee_triplet_homeostatic_plasticity_p95_changed_abs_delta` | `0.000825` |
| `video_l23ee_triplet_homeostatic_plasticity_max_abs_delta` | `0.009000` |
| `video_l23ee_triplet_homeostatic_plasticity_mean_gain_ratio` | `0.999687` |

The mass homeostasis audit for the full run was:
`incoming_mass_mean_ratio=0.999692`, `incoming_mass_min_ratio=0.748608`,
`incoming_mass_max_ratio=1.147847`, and
`incoming_mass_p95_abs_log_ratio=0.000000`. The tolerance row is diagnostic
only; the mechanism intentionally avoids exact per-cell or global
normalization.

## Biological Rationale

The rule is designed to match the direction of the biology without pretending
to be a complete dendritic or molecular model.

Ko et al. showed that nearby L2/3 pyramidal neurons in mouse V1 are more likely
to be connected when they have similar visual responses, supporting recurrent
functional specificity rather than random local recurrence. Cossell et al.
then showed that the strongest excitatory inputs to L2/3 cells are sparse and
concentrated among neurons with similar visual responses. Those two results
motivate the validation target: strong recurrent synapses should be enriched
among coactive/correlated L23E pairs, not merely moved by any plasticity rule.

Pfister and Gerstner motivate triplet-like spike-timing rules because pair-only
STDP does not capture frequency and higher-order spike effects. Froemke and Dan
showed that STDP under natural spike trains depends on preceding spike history,
which supports using traces rather than a single isolated pre/post pair.
Royer and Pare provide biological support for balancing synaptic potentiation
and depression to preserve total synaptic strength locally. Song et al. showed
highly nonrandom local cortical connectivity, including clustered and
heavy-tailed features, motivating the heavy-tail and shuffle controls.

Primary sources:

- Ko H, Hofer SB, Pichler B, Buchanan KA, Sjoestroem PJ, Mrsic-Flogel TD.
  2011. *Functional specificity of local synaptic connections in neocortical
  networks.* Nature 473, 87-91. DOI: https://doi.org/10.1038/nature09880
- Cossell L, Iacaruso MF, Muir DR, et al. 2015. *Functional organization of
  excitatory synaptic strength in primary visual cortex.* Nature 518, 399-403.
  DOI: https://doi.org/10.1038/nature14182
- Pfister JP, Gerstner W. 2006. *Triplets of spikes in a model of
  spike-timing-dependent plasticity.* Journal of Neuroscience 26, 9673-9682.
  DOI: https://doi.org/10.1523/JNEUROSCI.1425-06.2006
- Froemke RC, Dan Y. 2002. *Spike-timing-dependent synaptic modification
  induced by natural spike trains.* Nature 416, 433-438.
  DOI: https://doi.org/10.1038/416433a
- Royer S, Pare D. 2003. *Conservation of total synaptic weight through
  balanced synaptic depression and potentiation.* Nature 422, 518-522.
  DOI: https://doi.org/10.1038/nature01530
- Song S, Sjoestroem PJ, Reigl M, Nelson S, Chklovskii DB. 2005. *Highly
  nonrandom features of synaptic connectivity in local cortical circuits.*
  PLoS Biology 3, e68. DOI: https://doi.org/10.1371/journal.pbio.0030068

## Accepted Run Configuration

The accepted strict run used the `hold15625` matrix configuration on A6000.
Key settings:

| Setting | Value |
| --- | ---: |
| `video_l4_drive_scale` | `0.850000` |
| `final_post_video_assay_enabled` | `1.000000` |
| `video_ff_event_trace_enabled` | `1.000000` |
| `video_ff_event_trace_tau_pre_ms` | `20.000000` |
| `video_ff_event_trace_tau_post_ms` | `40.000000` |
| `video_ff_event_trace_tau_rate_ms` | `2000.000000` |
| `video_ff_event_trace_hetero_minus` | `0.000002` |
| `video_ff_event_trace_post_target_hz` | `0.050000` |
| `video_ff_event_trace_application_count` | `54` |
| `lower_v1_video_consolidation_heldout_fraction` | `0.156250` |
| `lower_v1_video_consolidation_frame_count` | `54` |
| `lower_v1_video_consolidation_heldout_excluded_frame_count` | `10` |
| `lower_v1_video_consolidation_future_frame_target_used` | `0.000000` |
| `lower_v1_video_consolidation_target_label_used` | `0.000000` |
| `lower_v1_video_consolidation_hva_feedback_enabled` | `0.000000` |
| `post_video_inhibitory_stabilization_pv_eta_scale` | `1.300000` |
| `video_pv_reliability_output_scale` | `1.050000` |
| `video_som_reliability_output_scale` | `0.900000` |

The accepted configuration uses `V1_VIDEO_CONSOLIDATION_HELDOUT_FRACTION=0.15625`
so the natural-video plastic-exposure frame count is truly 54 frames, with 10
held-out frames excluded. This mattered because the event-driven feedforward
plasticity spread gate was sensitive to the actual number of plastic exposure
frames.

## Strict Validator Metrics

From `.runs/logs/v1_triplet_l23ee_lr125_hold15625_acceptance_strict_validator.log`:

| Gate | Result |
| --- | --- |
| Natural-video L23E sparse safety | PASS: mean `0.415647 Hz`, p95 `2.500000 Hz`, p99 `8.125000 Hz`, frac<1 Hz `0.893678` |
| Raw L23E activity repeat stability | PASS: raw_oracle@5 `0.704167`, minimum `0.600000`, leaky ceiling `0.784375`, ceiling fraction `0.897742`, repeat corr `0.774457` |
| Event-driven FF plasticity | PASS: applications `54`, active edges `2191356`, changed fraction `0.051112`, p95 abs delta `0.000005`, max abs delta `0.003324` |
| Event-driven FF no-cheat | PASS: local only `1`, windowed count only `0`, future frame `0`, target label `0`, heldout frames `0`, HVA feedback `0` |
| Final-post-video OSI | PASS: source `final_post_video:l23e:core32`, full `0.853370`, control `0.000000`, delta `0.853370` |
| L23EE movement | PASS: active `643887`, changed fraction `0.069155`, p95 `0.000243`, bounds maintained |
| L23E full rates | PASS: median `0.026042 Hz`, frac<1 Hz `0.843750`, p99 `3.073438 Hz` |
| L23EE response-correlation specificity | PASS: active high-corr mean delta `0.000094`, active low-corr mean delta `0.000020`, active margin ok `1` |
| L23EE strong-synapse enrichment | PASS: top count `3826`, corr odds `1.184023`, combined coactive/co-tuned odds `1.272063`, top weight range `[0.007104,0.010000]` |
| L23EE heavy-tail-like distribution | PASS: gini `0.205810`, top1 mass `0.022244`, top10 mass `0.178914`, upper cap fraction `0.008984` |
| L23EE shuffle specificity | PASS: observed delta `0.037595`, shuffle q95 `0.014334`, z-score `4.543718` |
| L23EE recurrence contribution | PASS: mean corr on `0.194681`, off `0.184058`, delta `0.010623`; frac corr>0.2 delta `0.022053` |
| L23EE recurrence rate/OSI safety | PASS: mean peak on `8.384674`, off `8.168826`, peak ratio `0.974257`, mean OSI on `0.543200`, off `0.510154` |
| VIP weights | PASS: none found |

The strict acceptance log contains only PASS/INFO rows for the final gates
listed above and no `FAIL` row.

## Evidence Trail

The first triplet/homeostatic full run with learning rate `1.0` was not
accepted. It failed recurrent strong-synapse enrichment:

- Log:
  `.runs/logs/v1_triplet_l23ee_exact9_full_a6000_fullonly_recurrence_gate.log`
- Failure:
  `l23ee_strong_synapse_enrichment`
- Key values:
  `corr_odds_ratio=1.152786`, `combined_odds_ratio=1.184060`, below the
  required enrichment threshold.

Increasing only the triplet/homeostatic learning rate to `1.25` fixed the
recurrent enrichment gate without re-enabling old top-k competition:

- Log:
  `.runs/logs/v1_triplet_l23ee_lr125_exact9_full_a6000_smoke_validator_finalpost.log`
- Strong-synapse enrichment:
  `PASS`, `corr_odds_ratio=1.192381`, `combined_odds_ratio=1.214901`
- Raw oracle:
  `raw_oracle_at_k=0.653125`
- Validator command status:
  `VALIDATOR_COMMAND_STATUS=0`

The later strict acceptance pass required fixing the true natural-video
plastic-exposure frame count. With `V1_VIDEO_CONSOLIDATION_HELDOUT_FRACTION=0.15625`,
the run used `lower_v1_video_consolidation_frame_count=54` and excluded
10 held-out frames. This raised the event-driven FF spread above the strict
threshold while preserving recurrent enrichment and raw reliability.

Rejected targeted probes:

| Probe | Log | Result |
| --- | --- | --- |
| `tau25` | `.runs/logs/v1_triplet_l23ee_lr125_eventtau25_isolated_targeted_validator.log` | Rejected: event-driven FF changed fraction `0.045374 < 0.050000`; recurrent and raw gates passed, but event spread failed. |
| `event_repeat8` | `.runs/logs/v1_triplet_l23ee_lr125_eventrep8_isolated_targeted_validator.log` | Rejected: event-driven FF changed fraction `0.046024 < 0.050000`; strong-synapse combined odds `1.195962`, below the recurrent enrichment threshold. |
| `hetero_minus1e-6` | `.runs/logs/v1_triplet_l23ee_lr125_eventminus1e6_isolated_targeted_validator.log` | Rejected: event-driven FF changed fraction `0.046337 < 0.050000`; recurrent and raw gates passed, but event spread failed. |

The accepted targeted probe was:

- Log:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_isolated_targeted_validator.log`
- Result:
  `TARGETED_VALIDATOR_EXIT_STATUS=0`
- Key rows:
  event FF `changed_frac=0.051714`, `application_count=54`,
  recurrent enrichment `combined_odds_ratio=1.265694`, and
  raw_oracle@5 `0.718750`.

The matrix control needed explicit base-learning disables so the control did
not accidentally train through inherited environment defaults. In the accepted
control summary under
`.runs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full/`,
`video_ff_stdp_enabled=0`, `video_ff_event_trace_enabled=0`,
`video_l23ee_heterosyn_competition_active=0`, and
`video_l23ee_triplet_homeostatic_plasticity_active=0`.

## Hardcoded vs Emergent / No-Cheat Statement

Hardcoded or configured:

- The model architecture, sheet size, population counts, and validation core.
- The L4 simple-cell/Gabor-like video encoder and fixed L4 orientation scaffold.
- Local topographic wiring constraints and existing weight bounds.
- The chosen opt-in learning-rule constants and exposure schedule.
- The validator thresholds.

Emergent or activity-dependent in this branch:

- L23EE recurrent weight changes under the triplet/homeostatic rule.
- Enrichment of strong recurrent synapses among coactive/co-tuned L23E pairs.
- Heavy-tail-like recurrent weight distribution within bounds.
- Recurrence-dependent response-correlation contribution.
- Raw L23E repeat reliability during frozen natural-video evaluation.

No-cheat audit for the accepted recurrent rule:

- `video_l23ee_triplet_homeostatic_plasticity_orientation_label_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_future_frame_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_target_label_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_heldout_frames_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_validation_metric_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_global_rate_cap_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_global_normalization_used=0`
- `video_l23ee_triplet_homeostatic_plasticity_exact_normalization_used=0`

No-cheat audit for the accepted lower-V1 video exposure:

- `lower_v1_video_consolidation_heldout_frames_used=0`
- `lower_v1_video_consolidation_present_frame_drive_only=1`
- `lower_v1_video_consolidation_future_frame_target_used=0`
- `lower_v1_video_consolidation_target_label_used=0`
- `lower_v1_video_consolidation_hva_feedback_enabled=0`
- `video_ff_event_trace_future_frame_used=0`
- `video_ff_event_trace_target_label_used=0`
- `video_ff_event_trace_heldout_frames_used=0`
- `video_ff_event_trace_hva_feedback_enabled=0`

This is not a claim of full biological perfection. The implementation is still
a reduced point-neuron GeNN model with frame-binned video exposure, no explicit
dendritic compartments, no molecular calcium model, and no full cortical
developmental timescale. The claim supported here is narrower: replacing the
old ranked top-k recurrent competition with a local triplet/coactivity plus
soft postsynaptic homeostatic rule preserves the existing lower-V1 gates and
passes strict recurrent enrichment, heavy-tail, shuffle, recurrence-causality,
raw reliability, OSI, SOM, and rate-safety checks without using labels, future
frames, held-out frames, validation metrics, exact normalization, or global
rate caps.

## Artifact Paths

- Accepted strict validator:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_acceptance_strict_validator.log`
- Accepted full matrix run:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full.log`
- Accepted control run:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_control.log`
- Accepted SOM-off run:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_somoff.log`
- Accepted recurrence-off run:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_recoff.log`
- Accepted PV-dose runs:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_pvdose_100_a6000.log`,
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_pvdose_075_a6000.log`,
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_pvdose_050_a6000.log`,
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_pvdose_025_a6000.log`,
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_pvdose_000_a6000.log`
- Accepted targeted probe:
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_isolated_targeted_validator.log`
- Initial LR1.0 recurrent failure:
  `.runs/logs/v1_triplet_l23ee_exact9_full_a6000_fullonly_recurrence_gate.log`
- LR1.25 exact9 smoke pass:
  `.runs/logs/v1_triplet_l23ee_lr125_exact9_full_a6000_smoke_validator_finalpost.log`
- Rejected tau25 probe:
  `.runs/logs/v1_triplet_l23ee_lr125_eventtau25_isolated_targeted_validator.log`
- Rejected event-repeat8 probe:
  `.runs/logs/v1_triplet_l23ee_lr125_eventrep8_isolated_targeted_validator.log`
- Rejected hetero-minus1e-6 probe:
  `.runs/logs/v1_triplet_l23ee_lr125_eventminus1e6_isolated_targeted_validator.log`
- Accepted full summary:
  `.runs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full_summary.csv`
- Accepted L23EE specificity:
  `.runs/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full/v1_triplet_l23ee_lr125_hold15625_matrix_a6000_full_l23ee_specificity.csv`
