# Native Richter Task-Trained Next-V1-State Predictor - 2026-04-25

Training objective: predict the future lower-level V1_E trailer template
from context/leader only. No Q/activity/decoder/Richter dampening metric
is used in training or checkpoint selection.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_v1_state_prediction_20260425/stage1_ctx_pred_v1_template_seed42_n72.json`
Stage1 heldout eval: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_v1_state_prediction_20260425/stage1_ctx_pred_v1_template_seed42_n72_heldout_seed4243.json`
Sensory-only localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/task_prediction_v1_state_20260425/richter_dampening_stage1_v1template_n72_seed4242_reps4_rate100_sigma22_sensory_only_g0.json`
Thinning seeds: `1024`

## Frozen Evaluation Rows

| Row | Role | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | Beats previous all three | Pathology |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| balanced_g2_r03333333333333333 | minimum_required_preregistered_supported_balance | -307112.5056 | 17.8333 | 0.000336 | 0.004415 | 0.005442 | 0.006205 | False | none |
| analysis_only_g2p5_r0p5 | analysis_only_preregistered_sensitivity_row | -541985.5532 | 5.0000 | 0.000061 | -0.000692 | -0.001567 | -0.001953 | False | none |

## Constraints

- Explicit predicted-channel current scaling is disabled.
- V1 divisive suppressors are disabled.
- The frozen readout rows are preregistered; they are not selected by dampening metrics.
- The sigma22 sensory-only localizer is used only for the post-hoc pseudo-voxel transfer decoder.

## Baseline Comparator

Previous no-cheat baseline Q U-E: `21062.79`
Previous no-cheat baseline activity U-E: `22.33`
Previous no-cheat baseline decoder U-E @ keep_p=.02: `0.00548`
