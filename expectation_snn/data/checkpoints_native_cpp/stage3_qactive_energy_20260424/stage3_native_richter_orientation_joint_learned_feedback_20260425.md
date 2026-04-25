# Native Richter Joint Learned Direct+SOM Feedback - 2026-04-25

Training objective: original next-orientation/orientation-cell prediction plus
checkpointed H_pred->V1_E direct/apical and H_pred->V1_SOM routes learned
from future-state templates. Gain/balance selection uses validation
template-recruitment metrics only; no Q/activity/decoder/Richter gap metric
is used before the frozen test.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_joint_learned_feedback_20260425/stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed42_n72.json`
Heldout eval: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_joint_learned_feedback_20260425/stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed42_n72_heldout_seed4243.json`
Sensory-only localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/orientation_joint_learned_feedback_20260425/richter_dampening_orientation_joint_learned_seed4242_reps4_rate100_sigma22_sensory_only_g0.json`
Selected g_total/r: `128.0` / `1.0`

## Selection Rule

Before Richter testing, sweep joint learned direct+SOM g_total/r on validation seed 5353 with reps_expected=1/reps_unexpected=1; ignore expected/unexpected condition labels for scoring and do not inspect Q/activity/decoder gap metrics; keep rows with nonzero predicted V1 or SOM recruitment, V1_E population rate <= 100 Hz, V1_SOM population rate <= 50 Hz, and mean V1_E trailer count >= 5; select the eligible row maximizing V1 one-hot future-orientation cosine * log1p(predicted V1 count) plus SOM one-hot future-orientation cosine * log1p(predicted SOM count), tie-breaking toward lower total gain.

## Validation Grid

| g_total | r | Status | g_direct | g_som | V1 pred | V1 cos | SOM pred | SOM cos | V1 Hz | SOM Hz | Score | Eligible |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0 | PASS | 0 | 8 | 47.466667 | 0.155965 | 62.666667 | 0.496110 | 5.800000 | 0.813889 | 2.665953 | True |
| 8 | 0.333333 | PASS | 2 | 6 | 50.133333 | 0.154628 | 34.800000 | 0.520395 | 5.966667 | 0.362500 | 2.470319 | True |
| 8 | 1 | PASS | 4 | 4 | 56.000000 | 0.158096 | 0.000000 | 0.000000 | 6.333333 | 0.000000 | 0.639189 | True |
| 8 | 3 | PASS | 6 | 2 | 60.266667 | 0.160755 | 0.000000 | 0.000000 | 6.666667 | 0.000000 | 0.661547 | True |
| 8 | 1000 | PASS | 7.99201 | 0.00799201 | 60.800000 | 0.159823 | 0.000000 | 0.000000 | 6.933333 | 0.000000 | 0.659093 | True |
| 16 | 0 | PASS | 0 | 16 | 42.666667 | 0.156233 | 155.066667 | 0.450297 | 5.455556 | 2.485069 | 2.864155 | True |
| 16 | 0.333333 | PASS | 4 | 12 | 48.000000 | 0.157967 | 123.733333 | 0.457822 | 5.783333 | 1.930556 | 2.824312 | True |
| 16 | 1 | PASS | 8 | 8 | 53.333333 | 0.152034 | 84.400000 | 0.483956 | 6.572222 | 1.160764 | 2.759717 | True |
| 16 | 3 | PASS | 12 | 4 | 60.800000 | 0.158001 | 2.666667 | 0.089956 | 7.155556 | 0.015625 | 0.768461 | True |
| 16 | 1000 | PASS | 15.984 | 0.015984 | 61.333333 | 0.159218 | 0.000000 | 0.000000 | 7.327778 | 0.000000 | 0.657969 | True |
| 32 | 0 | PASS | 0 | 32 | 38.933333 | 0.159736 | 248.666667 | 0.415478 | 4.933333 | 4.670139 | 2.882470 | True |
| 32 | 0.333333 | PASS | 8 | 24 | 44.800000 | 0.154780 | 243.333333 | 0.429897 | 5.555556 | 4.359375 | 2.955727 | True |
| 32 | 1 | PASS | 16 | 16 | 48.000000 | 0.152503 | 185.066667 | 0.442914 | 6.300000 | 3.073264 | 2.908229 | True |
| 32 | 3 | PASS | 24 | 8 | 54.933333 | 0.153940 | 85.333333 | 0.478317 | 7.083333 | 1.194792 | 2.751922 | True |
| 32 | 1000 | PASS | 31.968 | 0.031968 | 61.333333 | 0.159211 | 0.000000 | 0.000000 | 7.483333 | 0.000000 | 0.657941 | True |
| 64 | 0 | PASS | 0 | 64 | 35.733333 | 0.164816 | 345.600000 | 0.404961 | 4.477778 | 6.879167 | 2.962224 | True |
| 64 | 0.333333 | PASS | 16 | 48 | 41.600000 | 0.158274 | 345.333333 | 0.400734 | 5.233333 | 6.977431 | 2.937073 | True |
| 64 | 1 | PASS | 32 | 32 | 44.266667 | 0.156249 | 289.200000 | 0.410828 | 5.883333 | 5.576736 | 2.925342 | True |
| 64 | 3 | PASS | 48 | 16 | 48.000000 | 0.153004 | 189.200000 | 0.439341 | 6.572222 | 3.162500 | 2.901161 | True |
| 64 | 1000 | PASS | 63.9361 | 0.0639361 | 61.333333 | 0.159211 | 0.000000 | 0.000000 | 7.516667 | 0.000000 | 0.657941 | True |
| 128 | 0 | PASS | 0 | 128 | 32.000000 | 0.170286 | 418.000000 | 0.379833 | 4.044444 | 9.195486 | 2.888788 | True |
| 128 | 0.333333 | PASS | 32 | 96 | 37.333333 | 0.164006 | 450.666667 | 0.381528 | 4.738889 | 9.879167 | 2.930279 | True |
| 128 | 1 | PASS | 64 | 64 | 39.466667 | 0.159833 | 409.866667 | 0.398350 | 5.255556 | 8.432292 | 2.988834 | True |
| 128 | 3 | PASS | 96 | 32 | 44.266667 | 0.156773 | 283.333333 | 0.409097 | 6.050000 | 5.485069 | 2.909168 | True |
| 128 | 1000 | PASS | 127.872 | 0.127872 | 61.333333 | 0.159211 | 0.000000 | 0.000000 | 7.522222 | 0.000000 | 0.657941 | True |

## Frozen Richter Test Rows

| Row | Role | Direct source | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| selected_joint_learned_g128_r1 | selected_joint_learned_direct_plus_som | learned | learned | 128 | 1 | -1221796.3728 | 129.8333 | 0.015066 | 0.046692 | 0.056559 | 0.067820 | False | none |
| selected_disabled_control_g128_r1 | selected_learned_routes_disabled_control | disabled | disabled | 128 | 1 | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| selected_shifted_control_g128_r1 | selected_wrong_hpred_mapping_control | learned-shifted | learned-shifted | 128 | 1 | -4189482.1756 | 87.8333 | 0.005259 | 0.030853 | 0.038401 | 0.045685 | False | none |
| selected_som_only_learned_g64_r0 | component_control_som_only_learned | disabled | learned | 64 | 0 | -2048606.8432 | 133.8333 | 0.051280 | 0.099080 | 0.106150 | 0.105174 | False | none |
| selected_direct_only_learned_g64p064_r1000 | component_control_direct_only_learned | learned | disabled | 64.064 | 1000 | -2216433.0996 | 15.6667 | -0.000417 | 0.000214 | -0.000193 | -0.000580 | False | none |

## Comparators

Original natural baseline: `{'q_u_minus_e': 21062.79, 'activity_u_minus_e': 22.33, 'decoder_u_minus_e_keep_p_0.02': 0.00548}`
Orientation learned-SOM gaincal g64: `{'q_u_minus_e': -2048606.843238, 'activity_u_minus_e': 133.833333, 'decoder_u_minus_e_keep_p_0.02': 0.09908}`
