# Native Richter Impartiality Factorial Assay - 2026-04-24

Primary scope: learned H_pred state plus ordinary static feedback routes only.
Explicit predicted suppression and divisive suppressors are disabled for all valid rows.
Raw-Ie predicted suppression, if listed, is an invalid engineered positive control.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix2/stage1_ctx_pred_seed42_n72.json`
Localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma22_sensory_only_feedback_g0_r03333333333333333_center010.json`
Thinning seeds: `1024`

## Result Table

| Condition | Row | Q E | Q U | Q U-E | Activity E | Activity U | Activity U-E | Decoder U-E kp=.02 | Decoder U-E kp=.015 | Pathology |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_held_replay_trailer_gating_only | held_replay_static_routes_g2_r03333333333333333 | 14680172.8396 | 14701235.6304 | 21062.7907 | 541.3333 | 563.6667 | 22.3333 | 0.005483 | 0.007233 | none |
| B_direct_apical_only | direct_only_r1000_g0.5 | 12401554.5276 | 12286184.0651 | -115370.4625 | 594.0000 | 579.1667 | -14.8333 | -0.002452 | -0.004588 | none |
| B_direct_apical_only | direct_only_r1000_g1 | 14584172.7094 | 14461806.6758 | -122366.0336 | 616.6667 | 591.8333 | -24.8333 | -0.004862 | -0.006775 | none |
| B_direct_apical_only | direct_only_r1000_g2 | 19133292.6561 | 18753089.8593 | -380202.7968 | 622.6667 | 599.5000 | -23.1667 | -0.005595 | -0.005900 | none |
| B_direct_apical_only | direct_only_r1000_g4 | 28247522.4263 | 27367635.9799 | -879886.4464 | 622.0000 | 605.6667 | -16.3333 | -0.004476 | -0.005300 | none |
| C_som_only | som_only_r0_g0.5 | 11077694.5502 | 11003006.4387 | -74688.1116 | 552.6667 | 565.6667 | 13.0000 | -0.001546 | -0.001851 | none |
| C_som_only | som_only_r0_g1 | 12006953.7853 | 11866877.6249 | -140076.1604 | 531.3333 | 559.0000 | 27.6667 | 0.004628 | 0.005900 | none |
| C_som_only | som_only_r0_g2 | 13802157.2784 | 13400417.8089 | -401739.4695 | 482.6667 | 547.3333 | 64.6667 | 0.022400 | 0.027130 | none |
| C_som_only | som_only_r0_g4 | 16769526.5441 | 16311930.0778 | -457596.4664 | 436.0000 | 526.0000 | 90.0000 | 0.031606 | 0.037821 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0 | 13802157.2784 | 13400417.8089 | -401739.4695 | 482.6667 | 547.3333 | 64.6667 | 0.022400 | 0.027130 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0.1 | 14049523.3625 | 13835999.0061 | -213524.3564 | 503.3333 | 549.3333 | 46.0000 | 0.007233 | 0.010915 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0.2 | 14373664.3727 | 14300741.5032 | -72922.8695 | 510.6667 | 555.1667 | 44.5000 | 0.010142 | 0.013560 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0.25 | 14499494.4463 | 14473679.6535 | -25814.7928 | 524.6667 | 558.3333 | 33.6667 | 0.008525 | 0.012543 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0.333333 | 14680172.8396 | 14701235.6304 | 21062.7907 | 541.3333 | 563.6667 | 22.3333 | 0.005483 | 0.007233 | none |
| D_direct_som_balance_fixed_total | balance_g2_r0.5 | 15277982.8077 | 15256200.6715 | -21782.1362 | 560.0000 | 572.3333 | 12.3333 | 0.003642 | 0.003215 | none |
| D_direct_som_balance_fixed_total | balance_g2_r1 | 16213266.3535 | 16131893.0147 | -81373.3388 | 588.6667 | 584.3333 | -4.3333 | -0.001180 | -0.003031 | none |
| D_direct_som_balance_fixed_total | balance_g2_r2 | 17306386.5041 | 17031345.8301 | -275040.6740 | 608.0000 | 593.3333 | -14.6667 | -0.003754 | -0.005961 | none |
| D_direct_som_balance_fixed_total | balance_g2_r10 | 18630371.5772 | 18270896.0531 | -359475.5241 | 622.6667 | 599.3333 | -23.3333 | -0.005564 | -0.005951 | none |
| D_direct_som_balance_fixed_total | balance_g2_r1000 | 19133292.6561 | 18753089.8593 | -380202.7968 | 622.6667 | 599.5000 | -23.1667 | -0.005595 | -0.005900 | none |

## Interpretation

Natural dampening found: `True`

Rows meeting all natural criteria:

- `held_replay_static_routes_g2_r03333333333333333`
- `balance_g2_r0.333333`

This is not a full fMRI forward model; the pseudo-voxel decoder is a
measurement-degraded population readout. Full argmax is reported only as
deterministic debug telemetry, not biological evidence.

## Artifacts

JSON summary: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/stage3_native_richter_impartiality_factorial_20260424.json`
- `held_replay_static_routes_g2_r03333333333333333`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_held_replay_static_routes_g2_r03333333333333333.json`
- `direct_only_r1000_g0.5`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_direct_only_r1000_g0p5.json`
- `direct_only_r1000_g1`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_direct_only_r1000_g1.json`
- `direct_only_r1000_g2`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_direct_only_r1000_g2.json`
- `direct_only_r1000_g4`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_direct_only_r1000_g4.json`
- `som_only_r0_g0.5`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_som_only_r0_g0p5.json`
- `som_only_r0_g1`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_som_only_r0_g1.json`
- `som_only_r0_g2`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_som_only_r0_g2.json`
- `som_only_r0_g4`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_som_only_r0_g4.json`
- `balance_g2_r0`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0.json`
- `balance_g2_r0.1`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0p1.json`
- `balance_g2_r0.2`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0p2.json`
- `balance_g2_r0.25`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0p25.json`
- `balance_g2_r0.333333`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0p333333.json`
- `balance_g2_r0.5`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r0p5.json`
- `balance_g2_r1`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r1.json`
- `balance_g2_r2`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r2.json`
- `balance_g2_r10`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r10.json`
- `balance_g2_r1000`: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/impartiality_factorial_20260424/richter_dampening_fix2_n72_seed4242_reps4_rate100_sigma22_balance_g2_r1000.json`
