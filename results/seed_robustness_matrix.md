# Seed-robustness paradigm matrix (Task #21)

Per-paradigm × per-net Δ_ex_unex at 3 seeds (42, 43, 44). Sign-stable means all 3 seeds agree on the sign of Δ (zeros tolerated). Δ_decC uses the shared decoder_c.pt; Δ_decA uses each ckpt's own joint-trained Dec A.

**Summary** — 8/17 rows sign-stable on Δ_decC, 15/17 on Δ_decA, 7/17 on both.

## Δ_decC (shared decoder_c.pt)

| # | Paradigm | Net | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Signs | Sign stable? |
|---|---|---|---:|---:|---:|:--:|:--:|
| 1 | HMM C1 (focused + HMM cue) | r1r2 | +0.0660 | +0.0540 | +0.0200 | +++ | yes |
| 2 | HMM C2 (routine + HMM cue) | r1r2 | +0.0290 | -0.0680 | -0.0970 | +−− | **NO** |
| 3 | HMM C3 (focused + zero cue) | r1r2 | +0.0410 | +0.0260 | +0.0150 | +++ | yes |
| 4 | HMM C4 (routine + zero cue) | r1r2 | +0.0360 | -0.0730 | -0.1410 | +−− | **NO** |
| 5 | HMM C1 (focused + HMM cue) | a1 | -0.0100 | -0.0130 | +0.0160 | −−+ | **NO** |
| 6 | HMM C1 (focused + HMM cue) | b1 | -0.0280 | -0.0320 | -0.0040 | −−− | yes |
| 7 | HMM C1 (focused + HMM cue) | c1 | -0.0090 | -0.0150 | -0.0190 | −−− | yes |
| 8 | HMM C1 (focused + HMM cue) | e1 | -0.0020 | -0.0110 | -0.0310 | −−− | yes |
| 9 | NEW (paired march) | r1r2 | +0.1254 | +0.1479 | +0.0854 | +++ | yes |
| 10 | M3R (matched_3row_ring) | r1r2 | -0.0274 | +0.0004 | -0.0683 | −+− | **NO** |
| 11 | HMS | r1r2 | +0.0525 | +0.1058 | -0.0461 | ++− | **NO** |
| 12 | HMS-T (tight-expected) | r1r2 | -0.0779 | +0.0669 | -0.1412 | −+− | **NO** |
| 13 | P3P (matched_probe_3pass) | r1r2 | +0.0513 | -0.0571 | -0.1250 | +−− | **NO** |
| 14 | VCD-test3 | r1r2 | -0.0708 | -0.0567 | -0.0810 | −−− | yes |
| 15 | M3R (modified: focused+march cue) | r1r2 | -0.0197 | -0.0487 | -0.0842 | −−− | yes |
| 16 | HMS-T (modified: focused+march cue) | r1r2 | +0.0510 | +0.0070 | -0.1682 | ++− | **NO** |
| 17 | VCD (modified: focused+march cue) | r1r2 | -0.0103 | +0.0063 | -0.0440 | −+− | **NO** |

## Δ_decA (each ckpt's joint-trained Dec A)

| # | Paradigm | Net | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Signs | Sign stable? |
|---|---|---|---:|---:|---:|:--:|:--:|
| 1 | HMM C1 (focused + HMM cue) | r1r2 | +0.3150 | +0.3440 | +0.3470 | +++ | yes |
| 2 | HMM C2 (routine + HMM cue) | r1r2 | +0.1620 | +0.1890 | +0.1720 | +++ | yes |
| 3 | HMM C3 (focused + zero cue) | r1r2 | +0.3120 | +0.3090 | +0.2990 | +++ | yes |
| 4 | HMM C4 (routine + zero cue) | r1r2 | +0.1700 | +0.1640 | +0.1520 | +++ | yes |
| 5 | HMM C1 (focused + HMM cue) | a1 | -0.0310 | -0.0040 | +0.0270 | −−+ | **NO** |
| 6 | HMM C1 (focused + HMM cue) | b1 | -0.0330 | +0.0070 | +0.0010 | −++ | **NO** |
| 7 | HMM C1 (focused + HMM cue) | c1 | +0.1770 | +0.1740 | +0.2090 | +++ | yes |
| 8 | HMM C1 (focused + HMM cue) | e1 | +0.1990 | +0.1810 | +0.2270 | +++ | yes |
| 9 | NEW (paired march) | r1r2 | +0.3871 | +0.4471 | +0.4304 | +++ | yes |
| 10 | M3R (matched_3row_ring) | r1r2 | -0.1548 | -0.0819 | -0.0646 | −−− | yes |
| 11 | HMS | r1r2 | -0.1865 | -0.0204 | -0.0631 | −−− | yes |
| 12 | HMS-T (tight-expected) | r1r2 | -0.3033 | -0.1036 | -0.1288 | −−− | yes |
| 13 | P3P (matched_probe_3pass) | r1r2 | +0.3846 | +0.4571 | +0.3500 | +++ | yes |
| 14 | VCD-test3 | r1r2 | -0.1666 | -0.1772 | -0.1096 | −−− | yes |
| 15 | M3R (modified: focused+march cue) | r1r2 | -0.1362 | -0.0805 | -0.0708 | −−− | yes |
| 16 | HMS-T (modified: focused+march cue) | r1r2 | -0.2937 | -0.1162 | -0.1545 | −−− | yes |
| 17 | VCD (modified: focused+march cue) | r1r2 | -0.0836 | -0.0722 | -0.0365 | −−− | yes |

## Δ_decC sign flips (paradigm × net cells where sign is NOT unanimous across seeds)

- Row 2: HMM C2 (routine + HMM cue) | r1r2 | signs across seeds = `+−−`
- Row 4: HMM C4 (routine + zero cue) | r1r2 | signs across seeds = `+−−`
- Row 5: HMM C1 (focused + HMM cue) | a1 | signs across seeds = `−−+`
- Row 10: M3R (matched_3row_ring) | r1r2 | signs across seeds = `−+−`
- Row 11: HMS | r1r2 | signs across seeds = `++−`
- Row 12: HMS-T (tight-expected) | r1r2 | signs across seeds = `−+−`
- Row 13: P3P (matched_probe_3pass) | r1r2 | signs across seeds = `+−−`
- Row 16: HMS-T (modified: focused+march cue) | r1r2 | signs across seeds = `++−`
- Row 17: VCD (modified: focused+march cue) | r1r2 | signs across seeds = `−+−`

## Δ_decA sign flips

- Row 5: HMM C1 (focused + HMM cue) | a1 | signs across seeds = `−−+`
- Row 6: HMM C1 (focused + HMM cue) | b1 | signs across seeds = `−++`

## Headline paradigms (Phase 4-7 mechanism map)

Cross-reference to docs/paradigm_sign_mechanism.md. Each row below is a paradigm whose Dec-C sign is the basis for a Mech 1 / Mech 2 verdict in that doc. The 'sign stable?' column tells us whether the verdict survives 3-seed perturbation.

| Paradigm | Net | Row | Phase 4-7 Mech (Dec C) | Δ_decC seed42 | seed43 | seed44 | Sign stable? |
|---|---|---:|---|---:|---:|---:|:--:|
| M3R native | r1r2 | 10 | Mech 1 | -0.0274 | +0.0004 | -0.0683 | **NO** |
| M3R modified | r1r2 | 15 | Mech 2 (weak) | -0.0197 | -0.0487 | -0.0842 | yes |
| HMS native | r1r2 | 11 | Mech 2 | +0.0525 | +0.1058 | -0.0461 | **NO** |
| HMS-T native | r1r2 | 12 | Mech 2 | -0.0779 | +0.0669 | -0.1412 | **NO** |
| HMS-T modified | r1r2 | 16 | Mech 2 | +0.0510 | +0.0070 | -0.1682 | **NO** |
| VCD-test3 native | r1r2 | 14 | Mech 1 | -0.0708 | -0.0567 | -0.0810 | yes |
| VCD-test3 modified | r1r2 | 17 | Mech 1 | -0.0103 | +0.0063 | -0.0440 | **NO** |
| HMM C1 (R1+R2) | r1r2 | 1 | Mech 1 (paired-fork V2-pred stratum) | +0.0660 | +0.0540 | +0.0200 | yes |
| HMM C1 (a1) | a1 | 5 | n/a (legacy) | -0.0100 | -0.0130 | +0.0160 | **NO** |
| HMM C1 (b1) | b1 | 6 | n/a (legacy) | -0.0280 | -0.0320 | -0.0040 | yes |
| HMM C1 (c1) | c1 | 7 | n/a (legacy) | -0.0090 | -0.0150 | -0.0190 | yes |
| HMM C1 (e1) | e1 | 8 | n/a (legacy) | -0.0020 | -0.0110 | -0.0310 | yes |
