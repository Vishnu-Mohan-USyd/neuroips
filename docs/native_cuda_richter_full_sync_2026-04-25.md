# Native CUDA Richter Full Sync Audit - 2026-04-25

## Scope

This report audits and syncs the native CUDA Richter work from the remote machine:

- Remote endpoint: `root@38.65.239.32 -p 23274 -i ~/.ssh/id_ed25519`
- Remote repo path: `/workspace/neuroips_gpu_migration_20260422/neuroips`
- Local repo path: `/home/vysoforlife/code_files/snn_v2_gpu/neuroips`
- Previous pushed branch: `cuda-richter-dampening-20260424`
- Previous pushed branch HEAD: `dfbdcabde6e77f42a2dbbbab9db0394c283c0214`
- Full-sync branch prepared locally: `cuda-richter-full-sync-20260425`

## Audit Finding

The previous GitHub branch was not the latest remote experimental code.

GitHub `cuda-richter-dampening-20260424` pointed at `dfbdcabde6e77f42a2dbbbab9db0394c283c0214`. The remote working tree was still on branch `ctx-pred-current-state-20260422` at `279892f22cdf055a6a2c23aeb5fe4f8d430b7beb`, with the native CUDA work present mostly as untracked files and modified tracked files. In particular, `cpp_cuda/`, `expectation_snn/cuda_sim/`, many native validators, and the later Richter analysis scripts were not represented by the remote Git HEAD.

The local sync copied the latest relevant remote code and selected artifacts into the local repository. A checksum audit over scoped code/docs/scripts showed:

- Remote scoped files: 123
- Local scoped files after sync: 127
- Missing remote files locally: 0
- Different remote/local checksums: 0
- Extra local-only files: prior local docs plus the interrupted, unvalidated `expectation_snn/scripts/run_native_richter_stage1_homeostasis_task_quality.py`

The unvalidated homeostasis runner is intentionally not part of the remote sync evidence.

## Sync Validation

Validation performed during the sync:

- Scoped local-vs-remote checksum comparison over native code, Python scripts, validation files, docs, and diagnostics: PASS.
- Staged text secret scan for common GitHub/OpenAI/AWS/private-key patterns: PASS.
- Python compile for synced Richter analysis scripts: PASS.
- Remote native build check: PASS.
- Remote build log: `/workspace/neuroips_gpu_migration_20260422/logs/full_sync_build_check_20260425.log`
- Remote `device-info`: CUDA device count `1`, GPU `NVIDIA GeForce RTX 4090`, runtime `12040`, driver `13000`.

## Synced Code Surfaces

The sync includes the native CUDA implementation and supporting Python surfaces that existed on the remote machine:

- `cpp_cuda/CMakeLists.txt`
- `cpp_cuda/include/expectation_snn_cuda/manifest.hpp`
- `cpp_cuda/src/bindings.cpp`
- `cpp_cuda/src/native_cli.cpp`
- `cpp_cuda/src/richter_eval.cu`
- `expectation_snn/cuda_sim/`
- `expectation_snn/scripts/analyze_native_richter_pseudovoxels.py`
- `expectation_snn/scripts/run_native_richter_impartiality_factorial.py`
- `expectation_snn/scripts/run_native_richter_readout_sensitivity.py`
- `expectation_snn/scripts/run_native_richter_task_prediction_v1_state.py`
- `expectation_snn/scripts/run_native_richter_learned_som_feedback.py`
- `expectation_snn/scripts/run_native_richter_learned_som_orientation_feedback.py`
- `expectation_snn/scripts/run_native_richter_learned_som_gain_calibration.py`
- `expectation_snn/scripts/run_native_richter_joint_learned_feedback.py`
- `expectation_snn/scripts/run_native_richter_prediction_efficiency_calibration.py`
- Native validation scripts under `expectation_snn/validation/`
- Diagnostic scripts under `scripts/`

The sync excludes build directories, logs, `__pycache__`, Python bytecode, secrets, and raw per-row Richter trial artifacts not needed because compact JSON/MD summaries were available.

## Architecture State

The current native path centers on `richter-dampening` in `cpp_cuda/src/native_cli.cpp` and `cpp_cuda/src/richter_eval.cu`. It supports:

- Frozen Stage1 `H_ctx -> H_pred` checkpoints.
- Static direct/apical `H_pred -> V1_E` feedback.
- Static center/surround `H_pred -> V1_SOM` feedback.
- Learned `H_pred -> V1_E` and `H_pred -> V1_SOM` feedback matrices checkpointed during prediction-only Stage1 training.
- Held H_pred replay telemetry and route controls.
- Q_active measurement fields for V1_E and V1_SOM.
- Pseudo-voxel readout utilities for post-hoc biological readout checks.
- Measurement-only V1_ERROR comparator populations in fixed and signed-normalized modes.

Explicit suppressors/divisive mechanisms also exist in the code for controlled experiments, but they are not valid evidence for emergent biological dampening when they directly encode predicted-channel suppression.

## Q_active Definition

Q_active is a measurement of integrated active charge in femtocoulombs, with `pA * ms` treated as fC.

For the native Richter path:

- V1_E: `abs(I_e) + abs(I_i) + abs(I_ap_e)`
- V1_SOM: `abs(I_e) + abs(I_i)`
- V1 total: V1_E plus V1_SOM

Q_active is reported per trial and by condition in native JSON summaries.

## Decoder Protocol

The deterministic full argmax decoder is debug-only. It is not biological evidence because it reads a noiseless 12-channel activity vector directly and tends to saturate.

The corrected pseudo-voxel decoder uses:

- 12 V1 channels pooled into 4 contiguous pseudo-voxels.
- Sensory-only localizer trials for training.
- 6-class actual-trailer nearest-centroid transfer decoding.
- Independent binomial thinning after voxel pooling.
- Keep probabilities `0.05`, `0.02`, `0.015`, and `0.01`.
- 1024 thinning seeds for robust final rows.
- Matched expected/unexpected trial counts by partitioning unexpected trials.
- Bootstrap confidence intervals and shuffled-label controls in robust analyses.

This is still a pseudo-voxel readout, not a full fMRI forward model.

## Valid Natural Baseline

The only clean natural row with the correct signs remains weak:

- Row: `held_replay_static_routes_g2_r03333333333333333`
- Q U-E: `+21062.79074941203`
- Activity U-E: `+22.333333333333258`
- Noisy pseudo-voxel decoder U-E at keep_p `0.02`: `+0.005482991536458351`
- Artifact: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_impartiality_factorial_20260424.json`
- Local mirror: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_impartiality_factorial_20260424.json`

This is valid because explicit predicted-current suppression and divisive suppressors are off. It is not strong enough to claim robust representational sharpening; it is a weak expectation-dampening sign pattern.

## Invalid Engineered Positive Control

The `raw_ie` predicted suppression mechanism produced strong metrics, but it is invalid as biological evidence because it hardcodes prediction-matched current suppression.

The final raw_ie positive-control row had:

- Scale: `16`
- Neighbor weight: `0.5`
- Q expected: `12130622.9738`
- Q unexpected: `13409791.7622`
- Activity expected: `311.3333`
- Activity unexpected: `449.8333`
- Noisy decoder U-E: keep_p `0.05 +0.065175`, `0.02 +0.104451`, `0.015 +0.111806`, `0.01 +0.111135`
- GPU mean runtime: about `3.69 s`
- CPU/GPU parity: passed
- Artifact: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_predicted_raw_suppression_sigma22_rollup_20260424.json`

This row should be treated only as an engineered positive control showing that the assay can detect a designed dampening mechanism.

## No-Cheat Experiments After Raw_IE

All later fair attempts failed to beat the weak natural baseline on all three metrics without explicit suppression.

### Readout Sensitivity Sweep

Static-route circuit readout sensitivity swept fixed feedback balances without changing training. It was analysis-only and not training evidence.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_readout_sensitivity_20260424.json`
- It found many decoder-positive rows, but Q was generally negative for stronger routes.
- Example: `gtotal1_r0p25` had Q U-E `-194246.82700665668`, activity U-E `+24.33333333333337`, decoder@0.02 `+0.0021158854166666843`.

### Next-V1-State Prediction Checkpoint

Prediction-only training for a next-V1-template target did not produce a better natural Richter effect.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_task_prediction_v1_state_20260425.json`
- `balanced_g2_r03333333333333333`: Q U-E `-307112.5056156069`, activity U-E `+17.83333333333337`, decoder@0.02 `+0.004414876302083351`.
- NO-GO because Q was negative and decoder did not beat the original weak baseline.

### Learned H_pred -> V1_SOM Feedback

The V1-template learned SOM route improved some decoder/activity measures but failed Q.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_learned_som_feedback_20260425.json`
- `learned_som_only_g2_r0`: Q U-E `-145995.23054567352`, activity U-E `+29.16666666666663`, decoder@0.02 `+0.006998697916666687`.
- NO-GO due negative Q.

### Orientation-Cell Learned SOM Feedback

The orientation-cell learned SOM route was wired but too weak at baseline gain.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_orientation_learned_som_feedback_20260425.json`
- `orientation_learned_som_plus_static_balanced_g2_r03333333333333333`: Q U-E `-92033.09985682368`, activity U-E `+8.833333333333371`, decoder@0.02 `-0.000946044921874984`.
- NO-GO.

### Learned SOM Gain Calibration

Gain selection used prediction/SOM recruitment metrics, not Richter dampening metrics. High-gain rows produced large activity/decoder effects but high expected-side current cost.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_orientation_learned_som_gain_calibration_20260425.json`
- `selected_gain_learned_som_only_g64_r0`: Q U-E `-2048606.8432376608`, activity U-E `+133.83333333333331`, decoder@0.02 `+0.09908040364583334`.
- NO-GO due strongly negative Q.

### Joint Learned Direct + SOM Feedback

Joint learned feedback also increased cost too much.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_orientation_joint_learned_feedback_20260425.json`
- `selected_joint_learned_g128_r1`: Q U-E `-1221796.3727861494`, activity U-E `+129.83333333333337`, decoder@0.02 `+0.04669189453125002`.
- NO-GO due negative Q.

### Prediction-Efficiency Calibration

Condition-blind efficiency calibration selected lower gains by prediction score per cost. It reduced the cost problem but still did not pass all metrics.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_prediction_efficiency_calibration_20260425.json`
- `efficiency_selected_som_g8_r0`: Q U-E `-91004.8534606807`, activity U-E `+39.0`, decoder@0.02 `+0.01055908203125002`.
- `efficiency_selected_joint_g8_r0`: same metrics as SOM-only because the selected balance resolved to SOM-only.
- NO-GO due negative Q.

### Task-Quality Stage1 Sweep

The clean next-orientation Stage1 sweep selected by heldout prediction quality only.

- Summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_orientation_task_quality_sweep_20260425.json`
- Selected checkpoint: `stage1_orientation_task_quality_sweep_20260425/stage1_ctx_pred_orientation_cell_taskquality_seed42_n72.json`
- Heldout metrics: forecast probability `0.8611111111111112`, context persistence `214 ms`, no-runaway `9.270833333333334 Hz`, template argmax `1.0`, template cosine `0.6202215117283464`.
- Fixed balanced Richter: Q U-E `-92033.09985681996`, activity U-E `+8.833333333333371`, decoder@0.02 `-0.000946044921874984`.
- NO-GO.

### Comparator / Error-Side Assays

Measurement-only V1_ERROR comparator populations were added to test an error-side readout without modifying V1_E/V1_SOM.

- Fixed comparator summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/comparator_v1_error_20260425/comparator_v1_error_rollup_20260425.json`
- Signed normalized comparator summary: `expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/comparator_v1_error_signed_20260425/comparator_v1_error_signed_normalized_rollup_20260425.json`

The signed normalized variant did not give a clean biological pass: signed total error activity showed an expected/unexpected gap, but Q moved in the wrong direction and shifted controls retained a substantial gap. It should not be claimed as V1 dampening.

## Included Artifacts

The sync includes compact summaries, rollups, and small Stage1 checkpoints needed to interpret or reproduce the reported rows. It intentionally excludes raw row-level trial artifacts and logs.

Examples of included summary artifacts:

- `stage3_native_richter_impartiality_factorial_20260424.json`
- `stage3_native_richter_readout_sensitivity_20260424.json`
- `stage3_native_richter_task_prediction_v1_state_20260425.json`
- `stage3_native_richter_learned_som_feedback_20260425.json`
- `stage3_native_richter_orientation_learned_som_feedback_20260425.json`
- `stage3_native_richter_orientation_learned_som_gain_calibration_20260425.json`
- `stage3_native_richter_orientation_joint_learned_feedback_20260425.json`
- `stage3_native_richter_prediction_efficiency_calibration_20260425.json`
- `stage3_orientation_task_quality_sweep_20260425.json`
- `comparator_v1_error_20260425/comparator_v1_error_rollup_20260425.json`
- `comparator_v1_error_signed_20260425/comparator_v1_error_signed_normalized_rollup_20260425.json`

## Current Scientific Conclusion

The repository now contains the full native CUDA experimental code and compact evidence trail from the remote machine. The honest result is:

- Engineered predicted-current suppression can create strong expected-lower-Q/activity/decoder effects, but it is not valid evidence for emergent biological dampening.
- The only valid no-cheat natural result is the weak fixed-route baseline.
- Learned feedback, task-prediction variants, readout sensitivity sweeps, efficiency calibration, and measurement-only comparator variants did not produce a stronger biologically valid pass.
- Future work should focus on prediction-task learning and local biological plasticity/homeostasis before Richter evaluation, not on test-metric optimization or hardcoded predicted-channel suppression.
