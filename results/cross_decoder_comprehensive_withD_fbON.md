# Cross-decoder comprehensive matrix (with Dec D)

## Long-form per-row table (raw ex/unex/Δ for each decoder)

| # | Assay | Network | n_ex | n_unex | ex_A | unex_A | Δ_A | ex_B | unex_B | Δ_B | ex_C | unex_C | Δ_C | ex_Dr | unex_Dr | Δ_Dr | ex_Ds | unex_Ds | Δ_Ds | sign-agree (ABC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | 0.6560 | 0.3410 | +0.3150 | 0.6030 | 0.5830 | +0.0200 | 0.6100 | 0.5440 | +0.0660 | 0.3690 | 0.3600 | +0.0090 | 0.3680 | 0.3350 | +0.0330 | ALL agree |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | 0.4490 | 0.2870 | +0.1620 | 0.4400 | 0.4740 | -0.0340 | 0.4780 | 0.4490 | +0.0290 | 0.3510 | 0.2990 | +0.0520 | 0.3600 | 0.2930 | +0.0670 | B outlier |
| 3 | HMM C3 (focused + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | 0.6080 | 0.2960 | +0.3120 | 0.5450 | 0.5310 | +0.0140 | 0.5400 | 0.4990 | +0.0410 | 0.3600 | 0.3360 | +0.0240 | 0.3540 | 0.2980 | +0.0560 | ALL agree |
| 4 | HMM C4 (routine + zero cue) | R1+R2 (emergent_seed42) | 1000 | 1000 | 0.4470 | 0.2770 | +0.1700 | 0.4370 | 0.4600 | -0.0230 | 0.4560 | 0.4200 | +0.0360 | 0.3340 | 0.2910 | +0.0430 | 0.3260 | 0.2780 | +0.0480 | B outlier |
| 5 | HMM C1 (focused + HMM cue) | a1 (legacy three-regimes) | 1000 | 1000 | 0.5740 | 0.6050 | -0.0310 | 0.5100 | 0.5160 | -0.0060 | 0.4900 | 0.5000 | -0.0100 | 0.4110 | 0.4620 | -0.0510 | 0.5450 | 0.5780 | -0.0330 | ALL agree |
| 6 | HMM C1 (focused + HMM cue) | b1 (legacy three-regimes) | 1000 | 1000 | 0.5780 | 0.6110 | -0.0330 | 0.4990 | 0.5220 | -0.0230 | 0.4850 | 0.5130 | -0.0280 | 0.4040 | 0.4550 | -0.0510 | 0.5680 | 0.5890 | -0.0210 | ALL agree |
| 7 | HMM C1 (focused + HMM cue) | c1 (legacy three-regimes) | 1000 | 1000 | 0.4370 | 0.2600 | +0.1770 | 0.4700 | 0.4500 | +0.0200 | 0.4350 | 0.4440 | -0.0090 | 0.3890 | 0.3200 | +0.0690 | 0.3850 | 0.3220 | +0.0630 | C outlier |
| 8 | HMM C1 (focused + HMM cue) | e1 (legacy three-regimes) | 1000 | 1000 | 0.4770 | 0.2780 | +0.1990 | 0.4950 | 0.4540 | +0.0410 | 0.4580 | 0.4600 | -0.0020 | 0.3790 | 0.3840 | -0.0050 | 0.3710 | 0.3630 | +0.0080 | C outlier |
| 9 | NEW (paired march, eval_ex_vs_unex_decC) | R1+R2 (emergent_seed42) | 2400 | 2400 | 0.7125 | 0.3254 | +0.3871 | 0.7346 | 0.6492 | +0.0854 | 0.7071 | 0.5817 | +0.1254 | 0.4529 | 0.3862 | +0.0667 | 0.4358 | 0.3708 | +0.0650 | ALL agree |
| 10 | M3R (matched_3row_ring) | R1+R2 (emergent_seed42) | 1084 | 3295 | 0.6965 | 0.8513 | -0.1548 | 0.8407 | 0.8513 | -0.0105 | 0.7399 | 0.7672 | -0.0274 | 0.4124 | 0.4039 | +0.0084 | 0.4723 | 0.4267 | +0.0456 | ALL agree |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 (emergent_seed42) | 3078 | 149 | 0.7934 | 0.9799 | -0.1865 | 0.8680 | 0.9793 | -0.1113 | 0.7908 | 0.7383 | +0.0525 | 0.4464 | 0.4430 | +0.0034 | 0.4574 | 0.4832 | -0.0258 | C outlier |
| 12 | HMS-T (matched_hmm_ring_sequence --tight-expected) | R1+R2 (emergent_seed42) | 791 | 105 | 0.6776 | 0.9810 | -0.3033 | 0.7709 | 0.9143 | -0.1434 | 0.6650 | 0.7429 | -0.0779 | 0.3944 | 0.4476 | -0.0532 | 0.4513 | 0.2857 | +0.1656 | ALL agree |
| 13 | P3P (matched_probe_3pass) | R1+R2 (emergent_seed42) | 39 | 39 | 0.6923 | 0.3077 | +0.3846 | 0.6857 | 0.6571 | +0.0286 | 0.8205 | 0.7692 | +0.0513 | 0.5128 | 0.2821 | +0.2308 | 0.5128 | 0.3846 | +0.1282 | ALL agree |
| 14 | VCD-test3 (v2_confidence_dissection) | R1+R2 (emergent_seed42) | 7981 | 7981 | 0.6003 | 0.7669 | -0.1666 | 0.5586 | 0.6398 | -0.0812 | 0.6336 | 0.7044 | -0.0708 | 0.4676 | 0.4106 | +0.0570 | 0.4697 | 0.4151 | +0.0546 | ALL agree |
| 15 | M3R (matched_3row_ring) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 3272 | 6484 | 0.7167 | 0.8529 | -0.1362 | 0.8217 | 0.8590 | -0.0372 | 0.7405 | 0.7602 | -0.0197 | 0.4282 | 0.4136 | +0.0145 | 0.5144 | 0.4346 | +0.0798 | ALL agree |
| 16 | HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 1143 | 111 | 0.6973 | 0.9910 | -0.2937 | 0.8649 | 0.9818 | -0.1169 | 0.7087 | 0.6577 | +0.0510 | 0.4724 | 0.3604 | +0.1121 | 0.4882 | 0.5135 | -0.0253 | C outlier |
| 17 | VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) | R1+R2 (emergent_seed42) | 6985 | 6985 | 0.7814 | 0.8650 | -0.0836 | 0.8368 | 0.8653 | -0.0285 | 0.7767 | 0.7870 | -0.0103 | 0.4773 | 0.4438 | +0.0335 | 0.5088 | 0.4329 | +0.0759 | ALL agree |

## Compact Δ side-by-side (5-decoder; sign-agreement is over A/B/C)

| # | Assay | Network | Δ_A | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | majority sign (ABC) | outlier (ABC) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 (emergent_seed42) | +0.3150 | +0.0200 | +0.0660 | +0.0090 | +0.0330 | + | — |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 (emergent_seed42) | +0.1620 | -0.0340 | +0.0290 | +0.0520 | +0.0670 | + | B |
| 3 | HMM C3 (focused + zero cue) | R1+R2 (emergent_seed42) | +0.3120 | +0.0140 | +0.0410 | +0.0240 | +0.0560 | + | — |
| 4 | HMM C4 (routine + zero cue) | R1+R2 (emergent_seed42) | +0.1700 | -0.0230 | +0.0360 | +0.0430 | +0.0480 | + | B |
| 5 | HMM C1 (focused + HMM cue) | a1 (legacy three-regimes) | -0.0310 | -0.0060 | -0.0100 | -0.0510 | -0.0330 | − | — |
| 6 | HMM C1 (focused + HMM cue) | b1 (legacy three-regimes) | -0.0330 | -0.0230 | -0.0280 | -0.0510 | -0.0210 | − | — |
| 7 | HMM C1 (focused + HMM cue) | c1 (legacy three-regimes) | +0.1770 | +0.0200 | -0.0090 | +0.0690 | +0.0630 | + | C |
| 8 | HMM C1 (focused + HMM cue) | e1 (legacy three-regimes) | +0.1990 | +0.0410 | -0.0020 | -0.0050 | +0.0080 | + | C |
| 9 | NEW (paired march, eval_ex_vs_unex_decC) | R1+R2 (emergent_seed42) | +0.3871 | +0.0854 | +0.1254 | +0.0667 | +0.0650 | + | — |
| 10 | M3R (matched_3row_ring) | R1+R2 (emergent_seed42) | -0.1548 | -0.0105 | -0.0274 | +0.0084 | +0.0456 | − | — |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 (emergent_seed42) | -0.1865 | -0.1113 | +0.0525 | +0.0034 | -0.0258 | − | C |
| 12 | HMS-T (matched_hmm_ring_sequence --tight-expected) | R1+R2 (emergent_seed42) | -0.3033 | -0.1434 | -0.0779 | -0.0532 | +0.1656 | − | — |
| 13 | P3P (matched_probe_3pass) | R1+R2 (emergent_seed42) | +0.3846 | +0.0286 | +0.0513 | +0.2308 | +0.1282 | + | — |
| 14 | VCD-test3 (v2_confidence_dissection) | R1+R2 (emergent_seed42) | -0.1666 | -0.0812 | -0.0708 | +0.0570 | +0.0546 | − | — |
| 15 | M3R (matched_3row_ring) (modified: focused+march cue) | R1+R2 (emergent_seed42) | -0.1362 | -0.0372 | -0.0197 | +0.0145 | +0.0798 | − | — |
| 16 | HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) | R1+R2 (emergent_seed42) | -0.2937 | -0.1169 | +0.0510 | +0.1121 | -0.0253 | − | C |
| 17 | VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) | R1+R2 (emergent_seed42) | -0.0836 | -0.0285 | -0.0103 | +0.0335 | +0.0759 | − | — |

## Per-decoder profile (A/B/C: ABC-sign-agreement stats; D_raw/D_shape: agreement with A/B/C majority)

| Decoder | n rows | mean |Δ| | max |Δ| | rows ALL agree (ABC) | rows agreeing w/ majority | rows disagreeing w/ majority | outlier / disagree rows |
|---|---|---|---|---|---|---|---|
| A | 17 | 0.2056 | 0.3871 | 11 | 17 | 0 | — |
| B | 17 | 0.0485 | 0.1434 | 11 | 15 | 2 | HMM C2 (routine + HMM cue) / R1+R2 (emergent_seed42), HMM C4 (routine + zero cue) / R1+R2 (emergent_seed42) |
| C | 17 | 0.0416 | 0.1254 | 11 | 13 | 4 | HMM C1 (focused + HMM cue) / c1 (legacy three-regimes), HMM C1 (focused + HMM cue) / e1 (legacy three-regimes), HMS (matched_hmm_ring_sequence) / R1+R2 (emergent_seed42), HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) / R1+R2 (emergent_seed42) |
| D_raw | 17 | 0.0520 | 0.2308 | — | 10 | 7 | HMM C1 (focused + HMM cue) / e1 (legacy three-regimes), M3R (matched_3row_ring) / R1+R2 (emergent_seed42), HMS (matched_hmm_ring_sequence) / R1+R2 (emergent_seed42), VCD-test3 (v2_confidence_dissection) / R1+R2 (emergent_seed42), M3R (matched_3row_ring) (modified: focused+march cue) / R1+R2 (emergent_seed42), HMS-T (matched_hmm_ring_sequence --tight-expected) (modified: focused+march cue) / R1+R2 (emergent_seed42), VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) / R1+R2 (emergent_seed42) |
| D_shape | 17 | 0.0585 | 0.1656 | — | 12 | 5 | M3R (matched_3row_ring) / R1+R2 (emergent_seed42), HMS-T (matched_hmm_ring_sequence --tight-expected) / R1+R2 (emergent_seed42), VCD-test3 (v2_confidence_dissection) / R1+R2 (emergent_seed42), M3R (matched_3row_ring) (modified: focused+march cue) / R1+R2 (emergent_seed42), VCD-test3 (v2_confidence_dissection) (modified: focused+march cue) / R1+R2 (emergent_seed42) |
