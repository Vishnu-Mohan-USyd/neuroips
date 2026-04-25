# Signed normalized V1_ERROR comparator rollup

Rollup JSON: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/comparator_v1_error_signed_20260425/comparator_v1_error_signed_normalized_rollup_20260425.json`

| row | total act E | total act U | U-E | total Q E | total Q U | U-E |
|---|---:|---:|---:|---:|---:|---:|
| comparator_off | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| signed_normalized | 1937.333333 | 2048.833333 | 111.500000 | 31733781.290817 | 31676771.879525 | -57009.411292 |
| signed_prediction_off | 2114.666667 | 2099.333333 | -15.333333 | 30922838.582484 | 30874276.491066 | -48562.091418 |
| signed_shifted_hpred | 1958.000000 | 2038.166667 | 80.166667 | 31733781.290816 | 31676771.879525 | -57009.411291 |

Verdict: `NO-GO`
Raw_ie/divisive suppressors were off; comparator does not feed back into V1_E/V1_SOM.
