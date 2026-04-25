# Native Richter Learned H_pred to V1_SOM Feedback - 2026-04-25

Training objective: future lower-level V1/SOM template prediction from
context/leader only. The learned feedback route is frozen before Richter
evaluation. Q/activity/decoder/dampening metrics are not used in training
or checkpoint selection.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_v1_state_learned_som_feedback_20260425/stage1_ctx_pred_v1_template_learned_som_seed42_n72.json`
Stage1 heldout eval: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_v1_state_learned_som_feedback_20260425/stage1_ctx_pred_v1_template_learned_som_seed42_n72_heldout_seed4243.json`
Sensory-only localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/learned_som_feedback_20260425/richter_dampening_learned_som_seed4242_reps4_rate100_sigma22_sensory_only_g0.json`
Thinning seeds: `1024`

## Frozen Rows

| Row | Role | SOM source | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats baseline all three | Pathology |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| learned_som_only_g2_r0 | primary_preregistered_learned_som_only | learned | -145995.2305 | 29.1667 | 0.000407 | 0.006999 | 0.009226 | 0.007141 | False | none |
| learned_som_shifted_control_g2_r0 | wrong_hpred_mapping_control | learned-shifted | -111516.4818 | 13.8333 | -0.000621 | -0.001109 | -0.001088 | 0.002482 | False | none |
| learned_som_disabled_control_g2_r0 | som_feedback_disabled_control | disabled | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| learned_som_plus_fixed_direct_g2_r03333333333333333 | analysis_only_preregistered_direct_plus_learned_som | learned | -248171.0776 | 15.8333 | -0.000142 | 0.003082 | 0.002380 | 0.004649 | False | none |

## Controls

The disabled control removes SOM feedback with the same checkpoint. The
learned-shifted control rotates H_pred source channels by +2 before applying
the learned SOM route, testing wrong-prediction targeting without trailer-label
access.

## Baseline Comparator

Original natural baseline Q U-E: `21062.79`
Original natural baseline activity U-E: `22.33`
Original natural baseline decoder U-E @ keep_p=.02: `0.00548`
