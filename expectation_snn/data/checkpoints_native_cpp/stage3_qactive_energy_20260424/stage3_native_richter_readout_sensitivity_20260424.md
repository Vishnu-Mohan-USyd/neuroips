# Native Richter Frozen-Task Static-Route Readout Sensitivity - 2026-04-24

This sweep uses a frozen Stage1 next-orientation task-trained checkpoint.
It does not train on Q/activity/decoder metrics and does not use explicit
predicted-channel suppression, divisive suppressors, or label leakage.

Checkpoint: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix2/stage1_ctx_pred_seed42_n72.json`
Localizer: `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage3_qactive_energy_20260424/richter_dampening_fix2_n72_seed4242_reps4_qactive_rate100_sigma22_sensory_only_feedback_g0_r03333333333333333_center010.json`
Rows: `40`
Thinning seeds per row: `1024`

## Ranked Rows

| Row | g_total | r | Q U-E | Activity U-E | Decoder U-E .05 | Decoder U-E .02 | Decoder U-E .015 | Decoder U-E .01 | All directional | Pathology |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gtotal2p5_r0p5 | 2.5 | 0.5 | 65022.9624 | 18.1667 | 0.000875 | 0.009155 | 0.009928 | 0.010895 | True | none |
| gtotal2_r0p333333333333 | 2.0 | 0.3333333333333333 | 21062.7907 | 22.3333 | 0.000651 | 0.005483 | 0.007233 | 0.011047 | True | none |
| gtotal2_r0p5 | 2.0 | 0.5 | -21782.1362 | 12.3333 | 0.000102 | 0.003642 | 0.003215 | 0.006694 | False | none |
| gtotal2_r0p25 | 2.0 | 0.25 | -25814.7928 | 33.6667 | 0.000641 | 0.008525 | 0.012543 | 0.014008 | False | none |
| gtotal3_r0p5 | 3.0 | 0.5 | -37670.9725 | 22.6667 | 0.000295 | 0.005890 | 0.006571 | 0.012431 | False | none |
| gtotal1p5_r0p45 | 1.5 | 0.45 | -54113.6137 | 9.1667 | 0.000102 | 0.003398 | 0.003448 | 0.005269 | False | none |
| gtotal1p5_r0p4 | 1.5 | 0.4 | -65335.9040 | 19.8333 | 0.001129 | 0.006378 | 0.006165 | 0.006165 | False | none |
| gtotal2p5_r0p4 | 2.5 | 0.4 | -75694.9463 | 22.8333 | 0.000977 | 0.005910 | 0.007334 | 0.011749 | False | none |
| gtotal2_r0p4 | 2.0 | 0.4 | -82539.8606 | 20.3333 | 0.001007 | 0.006104 | 0.008433 | 0.012665 | False | none |
| gtotal2_r0p3 | 2.0 | 0.3 | -96310.4223 | 27.1667 | 0.000661 | 0.008993 | 0.011546 | 0.013865 | False | none |
| gtotal1_r0p333333333333 | 1.0 | 0.3333333333333333 | -100910.6435 | 18.3333 | -0.000173 | 0.000977 | 0.000834 | 0.004150 | False | none |
| gtotal3_r0p28 | 3.0 | 0.28 | -110745.5549 | 35.8333 | 0.002879 | 0.016876 | 0.018646 | 0.020253 | False | none |
| gtotal1_r0p4 | 1.0 | 0.4 | -112524.2414 | 12.1667 | -0.000173 | 0.000153 | 0.000041 | 0.000488 | False | none |
| gtotal1_r0p3 | 1.0 | 0.3 | -114687.7975 | 21.6667 | 0.000295 | 0.003418 | 0.003703 | 0.006032 | False | none |
| gtotal3_r0p25 | 3.0 | 0.25 | -117602.6545 | 39.3333 | 0.003072 | 0.016001 | 0.018829 | 0.019450 | False | none |
| gtotal2_r0p36 | 2.0 | 0.36 | -122799.5316 | 21.3333 | 0.000549 | 0.006978 | 0.008189 | 0.010783 | False | none |
| gtotal1p5_r0p25 | 1.5 | 0.25 | -123978.0974 | 30.1667 | 0.001058 | 0.008036 | 0.009288 | 0.011119 | False | none |
| gtotal2p5_r0p36 | 2.5 | 0.36 | -126184.9035 | 26.8333 | 0.000539 | 0.007365 | 0.009216 | 0.013336 | False | none |
| gtotal1_r0p28 | 1.0 | 0.28 | -126625.1863 | 23.1667 | 0.000376 | 0.003764 | 0.005564 | 0.009135 | False | none |
| gtotal2_r0p28 | 2.0 | 0.28 | -129488.2080 | 28.3333 | 0.000722 | 0.007579 | 0.010508 | 0.014476 | False | none |
| gtotal1_r0p36 | 1.0 | 0.36 | -142910.7417 | 18.5000 | -0.000122 | -0.000122 | -0.000468 | 0.003245 | False | none |
| gtotal2_r0p45 | 2.0 | 0.45 | -151385.0925 | 18.1667 | 0.000814 | 0.006785 | 0.006592 | 0.009694 | False | none |
| gtotal1p5_r0p5 | 1.5 | 0.5 | -153764.8515 | 6.8333 | 0.000234 | 0.000417 | 0.001923 | 0.004008 | False | none |
| gtotal1_r0p45 | 1.0 | 0.45 | -157035.9527 | 10.3333 | 0.000448 | 0.004079 | 0.004761 | 0.001689 | False | none |
| gtotal1p5_r0p36 | 1.5 | 0.36 | -170046.5318 | 17.1667 | 0.001068 | 0.005992 | 0.006388 | 0.006724 | False | none |
| gtotal3_r0p3 | 3.0 | 0.3 | -172594.9944 | 30.5000 | 0.003194 | 0.014852 | 0.015686 | 0.017914 | False | none |
| gtotal1p5_r0p28 | 1.5 | 0.28 | -176912.4776 | 25.8333 | 0.001485 | 0.008118 | 0.009705 | 0.009277 | False | none |
| gtotal1_r0p25 | 1.0 | 0.25 | -194246.8270 | 24.3333 | 0.000092 | 0.002116 | 0.003876 | 0.007263 | False | none |
| gtotal1p5_r0p3 | 1.5 | 0.3 | -197530.4380 | 22.8333 | 0.000936 | 0.007345 | 0.011464 | 0.013458 | False | none |
| gtotal1p5_r0p333333333333 | 1.5 | 0.3333333333333333 | -209067.5182 | 20.0000 | 0.000854 | 0.007385 | 0.007751 | 0.011210 | False | none |
| gtotal1_r0p5 | 1.0 | 0.5 | -209520.2913 | 8.0000 | 0.000783 | 0.005066 | 0.005747 | 0.004476 | False | none |
| gtotal3_r0p333333333333 | 3.0 | 0.3333333333333333 | -227139.0847 | 33.5000 | 0.001444 | 0.012746 | 0.016164 | 0.020528 | False | none |
| gtotal2p5_r0p333333333333 | 2.5 | 0.3333333333333333 | -247464.8622 | 26.6667 | 0.000397 | 0.007792 | 0.009440 | 0.015798 | False | none |
| gtotal2p5_r0p3 | 2.5 | 0.3 | -282007.3762 | 27.8333 | 0.000834 | 0.009277 | 0.012929 | 0.015940 | False | none |
| gtotal2p5_r0p28 | 2.5 | 0.28 | -283289.7691 | 29.5000 | 0.000966 | 0.008128 | 0.008240 | 0.010549 | False | none |
| gtotal2p5_r0p45 | 2.5 | 0.45 | -283926.4708 | 21.5000 | 0.000712 | 0.005961 | 0.007690 | 0.012868 | False | none |
| gtotal3_r0p4 | 3.0 | 0.4 | -358802.9866 | 34.5000 | 0.001801 | 0.009969 | 0.013641 | 0.018127 | False | none |
| gtotal3_r0p45 | 3.0 | 0.45 | -422967.7961 | 30.5000 | 0.000529 | 0.011088 | 0.013784 | 0.019562 | False | none |
| gtotal2p5_r0p25 | 2.5 | 0.25 | -472423.2171 | 35.0000 | 0.001709 | 0.012421 | 0.014618 | 0.016927 | False | none |
| gtotal3_r0p36 | 3.0 | 0.36 | -510931.8430 | 37.1667 | 0.001444 | 0.014476 | 0.018412 | 0.021413 | False | none |

## Interpretation

Natural/static-route all-directional rows: `2`
Rows with positive decoder CI lower bound at all keep_p: `21`

This is a circuit readout sensitivity assay, not a training-selection loop.
Any promising setting must still be replicated across seeds/checkpoints
before being treated as robust emergent dampening.
