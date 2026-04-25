# Native Richter Orientation Learned-SOM Gain Calibration - 2026-04-25

Training objective: original next-orientation/orientation-cell prediction
plus checkpointed H_pred->V1_SOM route learned from future-state templates.
The gain is selected only from validation SOM-recruitment/template metrics.
No Q/activity/decoder/Richter expected-unexpected gap is used for training,
checkpoint selection, or gain selection.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_learned_som_gain_calibration_20260425/stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed42_n72.json`
Heldout eval: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_learned_som_gain_calibration_20260425/stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed42_n72_heldout_seed4243.json`
Sensory-only localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/orientation_learned_som_gain_calibration_20260425/richter_dampening_orientation_learned_som_gaincal_seed4242_reps4_rate100_sigma22_sensory_only_g0.json`
Selected gain: `64.0`

## Selection Rule

Before Richter testing, sweep learned SOM route g_total at r=0 on validation seed 5252 with reps_expected=1/reps_unexpected=1; ignore expected/unexpected condition labels and all Q/activity/decoder gap metrics; keep rows with predicted SOM count > 0, V1_SOM population rate <= 50 Hz, V1_E population rate <= 100 Hz, and mean V1_E trailer count >= 5; select the eligible gain maximizing mean(one-hot future-orientation SOM cosine) * log1p(mean predicted SOM count), tie-breaking toward lower gain.

## Validation Gain Table

| Gain | Status | Pred SOM count | SOM total | SOM cosine | Top1 pred | SOM rate Hz | Score | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | PASS | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.000000 | 0.000000 | False |
| 2 | PASS | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.000000 | 0.000000 | False |
| 4 | PASS | 0.666667 | 0.666667 | 0.033333 | 0.166667 | 0.001736 | 0.017028 | True |
| 8 | PASS | 59.733333 | 288.266667 | 0.505609 | 0.433333 | 0.750694 | 2.076278 | True |
| 16 | PASS | 144.933333 | 895.600000 | 0.446946 | 0.433333 | 2.332292 | 2.227197 | True |
| 32 | PASS | 245.333333 | 1755.200000 | 0.417663 | 0.466667 | 4.570833 | 2.299938 | True |
| 64 | PASS | 343.733333 | 2624.266667 | 0.405903 | 0.600000 | 6.834028 | 2.371596 | True |
| 128 | PASS | 410.933333 | 3366.400000 | 0.387342 | 0.300000 | 8.766667 | 2.332135 | True |
| 256 | PASS | 457.333333 | 4094.933333 | 0.364411 | 0.233333 | 10.663889 | 2.232961 | True |

## Frozen Richter Test Rows

| Row | Role | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| selected_gain_learned_som_only_g64_r0 | selected_gain_learned_som_only | learned | 64 | 0 | -2048606.8432 | 133.8333 | 0.051280 | 0.099080 | 0.106150 | 0.105174 | False | none |
| selected_gain_learned_som_plus_static_balanced_g64_r03333333333333333 | selected_gain_learned_som_plus_static_direct_balance | learned | 64 | 0.333333 | -1641709.2231 | 108.6667 | 0.018402 | 0.057393 | 0.064392 | 0.069010 | False | none |
| selected_gain_disabled_control_g64_r0 | selected_gain_som_feedback_disabled_control | disabled | 64 | 0 | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| selected_gain_shifted_control_g64_r0 | selected_gain_wrong_hpred_mapping_control | learned-shifted | 64 | 0 | -971867.1580 | 77.1667 | 0.019073 | 0.051666 | 0.056641 | 0.061615 | False | none |
| prior_gain_learned_som_only_g2_r0 | prior_orientation_learned_som_gain_baseline | learned | 2 | 0 | -66208.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |

## Comparators

Original natural baseline: `{'q_u_minus_e': 21062.79, 'activity_u_minus_e': 22.33, 'decoder_u_minus_e_keep_p_0.02': 0.00548}`
V1-template learned-SOM result: `{'q_u_minus_e': -145995.230546, 'activity_u_minus_e': 29.166667, 'decoder_u_minus_e_keep_p_0.02': 0.006999}`
Prior orientation learned-SOM g=2 result: `{'q_u_minus_e': -66208.09986, 'activity_u_minus_e': 10.166667, 'decoder_u_minus_e_keep_p_0.02': -0.000885}`
