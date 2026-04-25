# Native Richter Prediction-Efficiency Calibration - 2026-04-25

Scope: condition-blind gain calibration for existing prediction-only learned
feedback routes. No suppressors/divisive mechanisms are used. No expected-
vs-unexpected Q/activity/decoder gap metric is used for gain selection.

SOM checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_learned_som_gain_calibration_20260425/stage1_ctx_pred_orientation_cell_learned_som_gaincal_seed42_n72.json`
Joint checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_orientation_joint_learned_feedback_20260425/stage1_ctx_pred_orientation_cell_joint_learned_feedback_seed42_n72.json`

## Selection Formula

efficiency_score = prediction_score / max(mean_trial_v1_q_active_fC / 1e7, 1e-9); SOM family prediction_score = SOM one-hot future-orientation cosine * log1p(predicted SOM-channel count); joint family prediction_score = that SOM component plus V1_E one-hot future-orientation cosine * log1p(predicted V1_E-channel count). Eligibility requires nonzero prediction_score, V1_E population rate <= 100 Hz, V1_SOM population rate <= 50 Hz, and mean V1_E trailer count >= 5. Tie-break lower g_total.

## Selected Rows

SOM selected: `8.0`, r=`0.0`
Joint selected: `8.0`, r=`0.0`

## Validation Tradeoff

| Family | g_total | r | Status | Prediction score | Mean Q fC | Efficiency | V1 pred | SOM pred | V1 Hz | SOM Hz | Eligible |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| som | 2 | 0 | PASS | 0.000000 | 10966561.03 | 0.000000 | 51.200000 | 0.000000 | 5.905556 | 0.000000 | False |
| som | 4 | 0 | PASS | 0.017028 | 11625574.16 | 0.014647 | 51.200000 | 0.666667 | 5.905556 | 0.001736 | True |
| som | 8 | 0 | PASS | 2.059968 | 13138461.95 | 1.567891 | 46.933333 | 61.200000 | 5.688889 | 0.785417 | True |
| som | 16 | 0 | PASS | 2.242998 | 16072541.40 | 1.395547 | 43.733333 | 144.666667 | 5.327778 | 2.330903 | True |
| som | 32 | 0 | PASS | 2.284646 | 20890288.30 | 1.093640 | 37.333333 | 238.800000 | 4.877778 | 4.483333 | True |
| som | 64 | 0 | PASS | 2.366070 | 27620733.19 | 0.856628 | 32.533333 | 336.666667 | 4.333333 | 6.698264 | True |
| som | 128 | 0 | PASS | 2.299466 | 39300872.56 | 0.585093 | 29.866667 | 408.266667 | 3.944444 | 8.902778 | True |
| som | 256 | 0 | PASS | 2.215920 | 62251006.54 | 0.355965 | 26.666667 | 477.333333 | 3.427778 | 11.453472 | True |
| joint | 8 | 0 | PASS | 2.671108 | 13138461.95 | 2.033045 | 46.933333 | 61.200000 | 5.688889 | 0.785417 | True |
| joint | 8 | 0.333333 | PASS | 2.256310 | 13936662.34 | 1.618974 | 50.133333 | 28.666667 | 5.888889 | 0.314236 | True |
| joint | 8 | 1 | PASS | 0.707056 | 15487076.85 | 0.456546 | 58.666667 | 1.333333 | 6.222222 | 0.005208 | True |
| joint | 8 | 3 | PASS | 0.681479 | 16955174.36 | 0.401930 | 62.933333 | 0.000000 | 6.633333 | 0.000000 | True |
| joint | 8 | 1000 | PASS | 0.679631 | 18414895.23 | 0.369066 | 62.933333 | 0.000000 | 6.822222 | 0.000000 | True |
| joint | 16 | 0 | PASS | 2.865566 | 16072541.40 | 1.782895 | 43.733333 | 144.666667 | 5.327778 | 2.330903 | True |
| joint | 16 | 0.333333 | PASS | 2.826788 | 18243493.73 | 1.549477 | 49.066667 | 116.000000 | 5.722222 | 1.788194 | True |
| joint | 16 | 1 | PASS | 2.772783 | 21294850.90 | 1.302091 | 56.000000 | 77.200000 | 6.455556 | 1.042014 | True |
| joint | 16 | 3 | PASS | 0.813236 | 24768982.32 | 0.328328 | 62.400000 | 2.666667 | 7.005556 | 0.013889 | True |
| joint | 16 | 1000 | PASS | 0.686123 | 27660241.99 | 0.248054 | 63.466667 | 0.000000 | 7.150000 | 0.000000 | True |
| joint | 32 | 0 | PASS | 2.896744 | 20890288.30 | 1.386646 | 37.333333 | 238.800000 | 4.877778 | 4.483333 | True |
| joint | 32 | 0.333333 | PASS | 2.938475 | 25426801.74 | 1.155660 | 44.800000 | 223.466667 | 5.494444 | 4.000000 | True |
| joint | 32 | 1 | PASS | 2.913496 | 31590796.37 | 0.922261 | 49.600000 | 179.466667 | 6.216667 | 2.982292 | True |
| joint | 32 | 3 | PASS | 2.783877 | 38293269.03 | 0.726988 | 55.466667 | 86.666667 | 6.966667 | 1.204514 | True |
| joint | 32 | 1000 | PASS | 0.686123 | 44871745.36 | 0.152908 | 63.466667 | 0.000000 | 7.350000 | 0.000000 | True |
| joint | 64 | 0 | PASS | 2.969993 | 27620733.19 | 1.075277 | 32.533333 | 336.666667 | 4.333333 | 6.698264 | True |
| joint | 64 | 0.333333 | PASS | 2.980074 | 38395058.99 | 0.776161 | 40.000000 | 345.466667 | 5.133333 | 6.929167 | True |
| joint | 64 | 1 | PASS | 2.946665 | 48229375.24 | 0.610969 | 43.733333 | 284.666667 | 5.744444 | 5.458681 | True |
| joint | 64 | 3 | PASS | 2.933827 | 60477790.28 | 0.485108 | 49.600000 | 179.733333 | 6.550000 | 2.994097 | True |
| joint | 64 | 1000 | PASS | 0.686123 | 79640133.45 | 0.086153 | 63.466667 | 0.000000 | 7.422222 | 0.000000 | True |
| joint | 128 | 0 | PASS | 2.909503 | 39300872.56 | 0.740315 | 29.866667 | 408.266667 | 3.944444 | 8.902778 | True |
| joint | 128 | 0.333333 | PASS | 2.933355 | 59081070.65 | 0.496497 | 34.133333 | 443.333333 | 4.633333 | 9.732639 | True |
| joint | 128 | 1 | PASS | 2.965323 | 75900932.33 | 0.390683 | 37.333333 | 388.933333 | 5.238889 | 8.132639 | True |
| joint | 128 | 3 | PASS | 2.971981 | 97330014.46 | 0.305351 | 44.266667 | 282.533333 | 5.950000 | 5.356944 | True |
| joint | 128 | 1000 | PASS | 0.686123 | 149842088.51 | 0.045790 | 63.466667 | 0.000000 | 7.427778 | 0.000000 | True |

## Frozen Richter Test Rows

| Family | Row | Role | Direct source | SOM source | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats original all three | Pathology |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| som | efficiency_selected_som_g8_r0 | efficiency_selected_som_only | disabled | learned | 8 | 0 | -91004.8535 | 39.0000 | 0.001211 | 0.010559 | 0.013570 | 0.014964 | False | none |
| som | efficiency_selected_som_disabled_g8_r0 | selected_som_disabled_control | disabled | disabled | 8 | 0 | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| som | efficiency_selected_som_shifted_g8_r0 | selected_som_wrong_hpred_control | disabled | learned-shifted | 8 | 0 | -312030.5450 | 19.8333 | -0.000031 | 0.000793 | 0.003611 | 0.005575 | False | none |
| som | efficiency_prior_high_som_g64_r0 | prior_high_gain_som_comparator | disabled | learned | 64 | 0 | -2048606.8432 | 133.8333 | 0.051280 | 0.099080 | 0.106150 | 0.105174 | False | none |
| joint | efficiency_selected_joint_g8_r0 | efficiency_selected_joint_direct_plus_som | learned | learned | 8 | 0 | -91004.8535 | 39.0000 | 0.001211 | 0.010559 | 0.013570 | 0.014964 | False | none |
| joint | efficiency_selected_joint_disabled_g8_r0 | selected_joint_disabled_control | disabled | disabled | 8 | 0 | -16433.0999 | 10.1667 | -0.000498 | -0.000885 | -0.000671 | 0.000722 | False | none |
| joint | efficiency_selected_joint_shifted_g8_r0 | selected_joint_wrong_hpred_control | learned-shifted | learned-shifted | 8 | 0 | -312030.5450 | 19.8333 | -0.000031 | 0.000793 | 0.003611 | 0.005575 | False | none |
| joint | efficiency_prior_high_joint_g128_r1 | prior_high_gain_joint_comparator | learned | learned | 128 | 1 | -1221796.3728 | 129.8333 | 0.015066 | 0.046692 | 0.056559 | 0.067820 | False | none |

## Comparators

Original natural baseline: `{'q_u_minus_e': 21062.79, 'activity_u_minus_e': 22.33, 'decoder_u_minus_e_keep_p_0.02': 0.00548}`
SOM high-gain comparator: `{'q_u_minus_e': -2048606.843238, 'activity_u_minus_e': 133.833333, 'decoder_u_minus_e_keep_p_0.02': 0.09908}`
Joint high-gain comparator: `{'q_u_minus_e': -1221796.372786, 'activity_u_minus_e': 129.833333, 'decoder_u_minus_e_keep_p_0.02': 0.046692}`
