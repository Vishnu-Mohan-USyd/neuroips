# Native Richter Orientation-Cell Learned SOM Feedback - 2026-04-25

Training objective: original next-orientation/orientation-cell prediction
plus a prediction-only H_pred->V1_SOM route. No Q/activity/decoder/Richter
metrics are used for training or checkpoint selection.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_learned_som_feedback_20260425/stage1_ctx_pred_orientation_cell_learned_som_seed42_n72.json`
Stage1 heldout eval: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_learned_som_feedback_20260425/stage1_ctx_pred_orientation_cell_learned_som_seed42_n72_heldout_seed4243.json`
Sensory-only localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/orientation_learned_som_feedback_20260425/richter_dampening_orientation_learned_som_seed4242_reps4_rate100_sigma22_sensory_only_g0.json`
Thinning seeds: `1024`

## Frozen Rows

| Row | Role | SOM source | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| orientation_learned_som_only_g2_r0 | primary_preregistered_learned_som_only | learned | -66208.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| orientation_learned_som_plus_static_balanced_g2_r03333333333333333 | preregistered_learned_som_plus_static_direct_balance | learned | -92033.0999 | 8.8333 | 0.000163 | -0.000946 | -0.001465 | -0.001902 | False | none |
| orientation_learned_som_disabled_control_g2_r0 | som_feedback_disabled_control | disabled | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| orientation_learned_som_shifted_control_g2_r0 | wrong_hpred_mapping_control | learned-shifted | -66208.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |

## Comparators

Original natural baseline: `{'q_u_minus_e': 21062.79, 'activity_u_minus_e': 22.33, 'decoder_u_minus_e_keep_p_0.02': 0.00548}`
V1-template learned-SOM result: `{'q_u_minus_e': -145995.230546, 'activity_u_minus_e': 29.166667, 'decoder_u_minus_e_keep_p_0.02': 0.006999}`
