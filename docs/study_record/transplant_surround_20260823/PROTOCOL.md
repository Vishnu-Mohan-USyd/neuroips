# Weight-set strategy investigation (surround architecture) — protocol
Opened 2026-08-23 by team-lead. Status: Phase 1 (design) in flight.

## Goal (user-named)
Properly investigate the network's STRATEGY: which weight sets each network type (sharpening
α=0.0 vs dampening α=0.5, surround architecture s=0.04) actually utilizes to produce its
phenotype, and how. Methodology anchor named by the user: the transplant test — each trained
weight set transplanted individually and in combination into an untrained fresh host, phenotype
measured. Extend beyond the binary carry/doesn't-carry where informative (alignment/geometry of
each set, trained-vs-pretrain deltas, randomization controls).

## Baseline (established, original no-surround architecture, transplant_20260818 — frozen)
- Sharpening = co-adapted CELL+FB+GAINS trio; no partial combination works; misaligned trained
  FB is WORSE than untrained.
- Dampening = CELL+GAINS; untrained (random) FB suffices, 4/4 seeds (k applies negative gain to
  a magnitude-preserving random direction).
The new study must state, per regime, whether the surround architecture CHANGED this map.

## Weight sets (this architecture)
CELL (recurrent RNNCell 36→64), FB (W_fb Linear 64→36), GAINS (circ_raw → g0..g4 → k).
Fixed/non-weight: L4 encoder, surround kernel (config-carried; present in host by config).

## Design constraints
- Measurement-only: state-dict splicing + standard frozen assay. NO training runs.
- Donors: s=0.04 endpoints (both regimes), seed 8 primary; seeds 9/10/11 for confirmation of
  any load-bearing claim. Frozen run dirs READ-ONLY.
- Host convention, combination matrix (7 combos × regimes × seeds), FB controls
  (random / magnitude-matched / rotation-misaligned), and carry thresholds: inherited/adapted
  from transplant_20260818's frozen protocol — Phase 1 fixes them BEFORE any measurement and
  records them here. Any deviation from the original study's conventions must be stated with
  its reason.
- Pre-registered reading rules fixed in Phase 1: what effect fraction counts as "carries",
  "partial", "absent" — before results are seen.
- Envelope: cuda:0, MemAvailable ≥25 GB, sequential, one checkpoint at a time del+gc,
  PYTHONHASHSEED=0 python3 -B, writes only under /home/vishnu/scratch/transplant_surround_20260823/
  and this study dir.

## Phases
1. DESIGN (researcher, read-only): read transplant_20260818 protocol+results (frozen) and the
   surround study record; deliver DESIGN.md here — full run matrix, host convention, controls,
   carry thresholds, per-regime predictions from the established mechanism picture, and the
   deeper strategy analyses (weight deltas from pretrain, FB alignment geometry, gain
   configuration comparison) with exact measurement definitions.
2. EXECUTE (coder): implement splicing harness + run matrix + analyses; per-combination assay
   numbers; figures optional at lead's call.
3. VERIFY (validator): independent re-derivation of the load-bearing cells; GO/NO-GO on the
   strategy map.
4. Report to user: the strategy map per network type, original-vs-surround comparison, evidence.

## Log
- 2026-08-23: Protocol opened; Phase 1 dispatched to researcher.
- 2026-08-23: Phase 1 COMPLETE — DESIGN.md sha 3438e4c3…. Matrix 94 cells / 89 nets; conventions
  inherited verbatim from frozen originals; reading rules/floors/gates fixed pre-measurement;
  donors verified on disk. RECORD CORRECTION (researcher, lead-verified by grep on the frozen
  study): the original transplant study contains NO random/magnitude-matched/rotated FB controls
  anywhere — its dampening evidence is "PRETRAIN FB suffices" (and pretrain FB is
  task-informative, decode_A 0.74), so the prior "untrained/random FB suffices 4/4" gloss
  OVERSTATES the frozen record. This study's fresh FB controls close exactly that gap. Caveat
  logged: s=0.04 dampening donors seeds 9/10 are sub-band on M (O2) — within-seed carry ratios
  valid; any 8/11-vs-9/10 carry split is a reportable finding. Phase 2 dispatched to coder.
- 2026-08-23: G0 STOP + LEAD RULING. Coder's G0 tripped on seed-8 α0.5 anchors by ≤5.6e−17; the
  gate's own artifact check proved the measurement is BITWISE equal to both sha-pinned sources
  (stored eval report d01f88f9… AND validator v_ladder_s0p04.json); the DESIGN §3.6 anchor
  strings are 16-significant-digit transcriptions that parse 1 ulp off the artifact float64s
  (α0.0 strings happen to round-trip; α0.5 don't). RULING: authoritative G0/G-chain anchors =
  the cited ARTIFACT values at full precision, loaded programmatically from the sha-pinned
  files — never document-transcribed decimals (17 significant digits required for float64
  round-trip). DESIGN §3.6 stands as-written with this ruling superseding its literal decimals;
  no debugger needed (no unexplained discrepancy remains). Coder to reload anchors from
  artifacts, re-run N2 gates (expect abs diff 0.0), proceed.
- 2026-08-23: PHASE 2 COMPLETE (89 nets, all gates PASS incl. post-run G6 re-verify 48/48; one
  disclosed mid-run correction: floor-aware N5 extras selection, both runs' provenance kept).
  HEADLINES (pre-verification): DAMPENING — TPT (CELL+GAINS, pretrain FB) carries 4/4
  (ρ_M 0.927–0.986); **H-C1 REFUTED — no FB control carries on any seed** (TNT/TQT/TRT ρ_M
  0.46–0.61): dampening needs a TASK-MEANINGFUL FB direction, magnitude alone insufficient —
  corrects the prior "random FB suffices" gloss. GAINS-removal overshoot (Q2) repeats 4/4.
  SHARPENING — flank ratio unreadable 4/4 (host flank already at TTT level; registered fallback
  applied); TPT hit partial 0.36–0.54 (predicted band); FB placement premium PERSISTS (kernel
  did not absorb FB's role); H-C2 CONFIRMED (misaligned FB drives hit BELOW baseline,
  ρ −0.75..−0.88); NOVEL center/placement dissociation (GAINS alone transplant center, not
  hit); flank suppression = kernel work given well-placed f (TTP Δflank(s−s0) −0.150 vs PTP
  −0.063; PTP alone NO suppression). A_align: sharpening more alignment-critical 3/3
  resolvable. E_proj INVERTS prediction (dampening's small FB tweak sits in Δ_hh top-subspace;
  sharpening's big rewrite spread). α0.0 factorial trip-fragile at s=0.04 (new trips). No
  systematic 8/11-vs-9/10 split. Phase 3 dispatched to validator (load-bearing cells §2.5).
- 2026-08-23: PHASE 3 COMPLETE — **GO** (VERDICT.md sha ee83b5ed…, validator artifacts
  neuroips_outputs/validator_transplant_20260823T065540Z/). Validator rebuilt 46 distinct nets /
  50 table cells end-to-end from raw donor checkpoints with its own splicing + own measurement:
  hybrid state shas AND all endpoints bitwise == n4 50/50; E0 donor anchor 8/8 bitwise; own
  partition proof 8/8; ρ re-derived, max |Δρ| 8.9e−16, band agreement 100%; FB controls rebuilt
  from pinned RNG bitwise == controls_fb.pt; s→0, k tables (incl. recomputed no-surround
  originals), A_align, E_proj all bitwise/0.0-diff. H-C1 refutation and H-C2 confirmation
  verified on validator's own cells; R1 flank-floor fired genuinely 4/4; Q2 overshoot and PPT
  dissociation reproduced. N5 mid-run correction RULED registered-fallback-consistent, not
  post-hoc (α0.0 descriptive-set caveat recorded in VERDICT.md). Read-only conduct proven.
  NOTES (append-only corrections to the Phase-2 headline above): dampening TPT ρ_M range is
  0.927–1.004 (s11 = 1.0044, a carry; "0.986" was a transcription slip); FB placement premium
  persists 3/4 seeds above the original range, s9 = 0.464 sits below it.
- 2026-08-23: PHASE 4 — strategy-map report delivered to user. STUDY CLOSED.
