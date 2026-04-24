# 17-row cross-decoder matrix (7 decoders: A, A', B, C, D-raw, D-shape, E)

Dec E trained per-network on frozen fully-trained L2/3 with natural HMM stream (task_p_switch per each ckpt's yaml — Markov p=0.2 for R1+R2, Bernoulli-per-batch for legacy configs). R1+R2 Dec E ran 5000 steps; legacy a1/b1/c1 Dec E training hit a post-training bug at step 5000 so ckpts are **step-4000-recovered** (val_acc within ~0.02 of projected step-5000). e1 Dec E is full 5000 steps.

A' column only populated for R1+R2 rows (Dec A' was trained on R1+R2).

| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A' | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | Δ_E | ABC sign-agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.3150 | +0.3070 | +0.0200 | +0.0660 | +0.0090 | +0.0330 | +0.3290 | ALL (+) |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.1620 | +0.1690 | -0.0340 | +0.0290 | +0.0520 | +0.0670 | +0.1510 | B-out |
| 3 | HMM C3 (focused + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.3120 | +0.2820 | +0.0140 | +0.0410 | +0.0240 | +0.0560 | +0.2750 | ALL (+) |
| 4 | HMM C4 (routine + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.1700 | +0.1560 | -0.0230 | +0.0360 | +0.0430 | +0.0480 | +0.1580 | B-out |
| 5 | HMM C1 (focused + HMM cue) | a1 (legacy three-regimes) | 1000 | 1000 | -0.0310 | -0.0220 | -0.0060 | -0.0100 | -0.0510 | -0.0330 | +0.0400 | ALL (−) |
| 6 | HMM C1 (focused + HMM cue) | b1 (legacy three-regimes) | 1000 | 1000 | -0.0330 | -0.0320 | -0.0230 | -0.0280 | -0.0510 | -0.0210 | +0.0240 | ALL (−) |
| 7 | HMM C1 (focused + HMM cue) | c1 (legacy three-regimes) | 1000 | 1000 | +0.1770 | +0.1870 | +0.0200 | -0.0090 | +0.0690 | +0.0630 | +0.1800 | C-out |
| 8 | HMM C1 (focused + HMM cue) | e1 (legacy three-regimes) | 1000 | 1000 | +0.1990 | +0.2130 | +0.0410 | -0.0020 | -0.0050 | +0.0080 | +0.2300 | C-out |
| 9 | NEW (paired march, eval_ex_vs_unex_decC) | R1+R2 (emergent_seed42) | 2400 | 2400 | +0.3871 | +0.3888 | +0.0854 | +0.1254 | +0.0667 | +0.0650 | +0.3888 | ALL (+) |
| 10 | M3R (matched_3row_ring) | R1+R2 (emergent_seed42) | 1084 | 3295 | -0.1548 | -0.0887 | -0.0105 | -0.0274 | +0.0084 | +0.0456 | -0.0799 | ALL (−) |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 (emergent_seed42) | 3078 | 149 | -0.1865 | -0.1556 | -0.1113 | +0.0525 | +0.0034 | -0.0258 | -0.1600 | C-out |
| 12 | HMS-T (matched_hmm_ring_sequence --tight-expected) | R1+R2 (emergent_seed42) | 791 | 105 | -0.3033 | -0.2112 | -0.1434 | -0.0779 | -0.0532 | +0.1656 | -0.2074 | ALL (−) |
| 13 | P3P (matched_probe_3pass) | R1+R2 (emergent_seed42) | 39 | 39 | +0.3846 | +0.3902 | +0.0286 | +0.0513 | +0.2308 | +0.1282 | +0.4359 | ALL (+) |
| 14 | VCD-test3 (v2_confidence_dissection) | R1+R2 (emergent_seed42) | 7981 | 7981 | -0.1666 | -0.1987 | -0.0812 | -0.0708 | +0.0570 | +0.0546 | -0.2006 | ALL (−) |
| 15 | M3R (matched_3row_ring) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 3272 | 6484 | -0.1362 | -0.1122 | -0.0372 | -0.0197 | +0.0145 | +0.0798 | -0.1053 | ALL (−) |
| 16 | HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 1143 | 111 | -0.2937 | -0.2031 | -0.1169 | +0.0510 | +0.1121 | -0.0253 | -0.2061 | C-out |
| 17 | VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 6985 | 6985 | -0.0836 | -0.1165 | -0.0285 | -0.0103 | +0.0335 | +0.0759 | -0.1165 | ALL (−) |

## Per-decoder magnitude profile

| Decoder | n rows with Δ | mean \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|
| A | 17 | +0.2056 | +0.3871 |
| A_prime | 17 | +0.1902 | +0.3902 |
| B | 17 | +0.0485 | +0.1434 |
| C | 17 | +0.0416 | +0.1254 |
| D_raw | 17 | +0.0520 | +0.2308 |
| D_shape | 17 | +0.0585 | +0.1656 |
| E | 17 | +0.1934 | +0.4359 |

## Dec E sign flips vs Dec A (R1+R2 rows)

- **HMM C1 (focused + HMM cue)**: Δ_A -0.0310 vs Δ_E +0.0400
- **HMM C1 (focused + HMM cue)**: Δ_A -0.0330 vs Δ_E +0.0240

## Dec E sign flips vs Dec A' (R1+R2 rows)

- **HMM C1 (focused + HMM cue)**: Δ_A' -0.0220 vs Δ_E +0.0400
- **HMM C1 (focused + HMM cue)**: Δ_A' -0.0320 vs Δ_E +0.0240
