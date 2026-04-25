# 17-row cross-decoder matrix with 20k Dec A' (Task #7 Part B)

**Design choice.** This matrix REPLACES the previous Dec A' column (R1+R2-only, 5k Adam) with a new column trained at 20 000 Adam steps per-net across all 5 nets (r1r2 + a1/b1/c1/e1). The 5k Dec A' values are archived as `decA_prime_5k_delta` for reference. The 20k Dec A' is trained with the EXACT same protocol as 5k (lr 1e-3 Adam, seed 42, batch 32, seq 25, 50/50 task_state) — only `--n-steps 5000 → 20000`. Per-net 10k HMM stratified top-1 (Task #2 convention): r1r2 0.5729, a1 0.6709, b1 0.6625, c1 (Task #7 Part A new), e1 (Task #7 Part A new).

**Sign-flip analysis is the headline question.** Does the 5k Dec A' positive sign on rows 5/6 (a1/b1 HMM C1) — which was the canonical evidence for the now-retracted 'Dec A vs Dec E dissociation' — PERSIST when Dec A' is properly optimised at 20k steps, or does it COLLAPSE to agreement with Dec A's negative sign? See § 'Rows 5/6 persistence test' below for the answer.

| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A'(20k) | Δ_A'(5k) | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | Δ_E | ABC sign-agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.3150 | +0.3190 | +0.3070 | +0.0200 | +0.0660 | +0.0090 | +0.0330 | +0.3290 | ALL (+) |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.1620 | +0.1740 | +0.1690 | -0.0340 | +0.0290 | +0.0520 | +0.0670 | +0.1510 | B-out |
| 3 | HMM C3 (focused + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.3120 | +0.2970 | +0.2820 | +0.0140 | +0.0410 | +0.0240 | +0.0560 | +0.2750 | ALL (+) |
| 4 | HMM C4 (routine + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | +0.1700 | +0.1770 | +0.1560 | -0.0230 | +0.0360 | +0.0430 | +0.0480 | +0.1580 | B-out |
| 5 | HMM C1 (focused + HMM cue) | a1 (legacy three-regimes) | 1000 | 1000 | -0.0310 | +0.2100 | -0.0220 | -0.0060 | -0.0100 | -0.0510 | -0.0330 | +0.0400 | ALL (−) |
| 6 | HMM C1 (focused + HMM cue) | b1 (legacy three-regimes) | 1000 | 1000 | -0.0330 | +0.1800 | -0.0320 | -0.0230 | -0.0280 | -0.0510 | -0.0210 | +0.0240 | ALL (−) |
| 7 | HMM C1 (focused + HMM cue) | c1 (legacy three-regimes) | 1000 | 1000 | +0.1770 | +0.2490 | +0.1870 | +0.0200 | -0.0090 | +0.0690 | +0.0630 | +0.1800 | C-out |
| 8 | HMM C1 (focused + HMM cue) | e1 (legacy three-regimes) | 1000 | 1000 | +0.1990 | +0.2580 | +0.2130 | +0.0410 | -0.0020 | -0.0050 | +0.0080 | +0.2300 | C-out |
| 9 | NEW (paired march, eval_ex_vs_unex_decC) | R1+R2 (emergent_seed42) | 2400 | 2400 | +0.3871 | +0.3850 | +0.3888 | +0.0854 | +0.1254 | +0.0667 | +0.0650 | +0.3888 | ALL (+) |
| 10 | M3R (matched_3row_ring) | R1+R2 (emergent_seed42) | 1084 | 3295 | -0.1548 | -0.0550 | -0.0887 | -0.0105 | -0.0274 | +0.0084 | +0.0456 | -0.0799 | ALL (−) |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 (emergent_seed42) | 3078 | 149 | -0.1865 | -0.1697 | -0.1556 | -0.1113 | +0.0525 | +0.0034 | -0.0258 | -0.1600 | C-out |
| 12 | HMS-T (matched_hmm_ring_sequence --tight-expected) | R1+R2 (emergent_seed42) | 791 | 105 | -0.3033 | -0.2325 | -0.2112 | -0.1434 | -0.0779 | -0.0532 | +0.1656 | -0.2074 | ALL (−) |
| 13 | P3P (matched_probe_3pass) | R1+R2 (emergent_seed42) | 39 | 39 | +0.3846 | +0.4359 | +0.3902 | +0.0286 | +0.0513 | +0.2308 | +0.1282 | +0.4359 | ALL (+) |
| 14 | VCD-test3 (v2_confidence_dissection) | R1+R2 (emergent_seed42) | 7981 | 7981 | -0.1666 | -0.2006 | -0.1987 | -0.0812 | -0.0708 | +0.0570 | +0.0546 | -0.2006 | ALL (−) |
| 15 | M3R (matched_3row_ring) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 3272 | 6484 | -0.1362 | -0.1194 | -0.1122 | -0.0372 | -0.0197 | +0.0145 | +0.0798 | -0.1053 | ALL (−) |
| 16 | HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 1143 | 111 | -0.2937 | -0.2124 | -0.2031 | -0.1169 | +0.0510 | +0.1121 | -0.0253 | -0.2061 | C-out |
| 17 | VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 6985 | 6985 | -0.0836 | -0.1254 | -0.1165 | -0.0285 | -0.0103 | +0.0335 | +0.0759 | -0.1165 | ALL (−) |

## Rows 5/6 persistence test — does 5k Dec A' positive sign on a1/b1 HMM C1 collapse to negative at 20k?

### a1 (legacy three-regimes)

- Δ_A (Dec A original):       -0.0310 (−)
- Δ_A' (5k Adam, original):   -0.0220 (−)
- Δ_A' (20k Adam, NEW):       +0.2100 (+)
- Sign collapse 5k → 20k:     Yes (5k -> 20k changed sign)
- 20k matches Dec A sign:     No

### b1 (legacy three-regimes)

- Δ_A (Dec A original):       -0.0330 (−)
- Δ_A' (5k Adam, original):   -0.0320 (−)
- Δ_A' (20k Adam, NEW):       +0.1800 (+)
- Sign collapse 5k → 20k:     Yes (5k -> 20k changed sign)
- 20k matches Dec A sign:     No


## Per-decoder magnitude + ABC-majority agreement profile (all 17 rows)

| Decoder | n rows with Δ | mean \|Δ\| | max \|Δ\| | agrees with ABC majority |
|---|---:|---:|---:|---|
| A | 17 | +0.2056 | +0.3871 | — |
| A_prime_5k | 17 | +0.1902 | +0.3902 | 17/17 |
| A_prime_20k | 17 | +0.2235 | +0.4359 | 15/17 |
| B | 17 | +0.0485 | +0.1434 | 15/17 |
| C | 17 | +0.0416 | +0.1254 | 13/17 |
| D_raw | 17 | +0.0520 | +0.2308 | 10/17 |
| D_shape | 17 | +0.0585 | +0.1656 | 12/17 |
| E | 17 | +0.1934 | +0.4359 | 15/17 |

**ABC ALL-agree row count:** 11/17.


## Sign flips: Dec A vs 20k Dec A' (per-row)

| Row | Assay | Network | Δ_A | Δ_A'(20k) | Δ_A'(5k) |
|---:|---|---|---:|---:|---:|
| 5 | HMM C1 (focused + HMM cue) | a1 (legacy three-regimes) | -0.0310 | +0.2100 | -0.0220 |
| 6 | HMM C1 (focused + HMM cue) | b1 (legacy three-regimes) | -0.0330 | +0.1800 | -0.0320 |
