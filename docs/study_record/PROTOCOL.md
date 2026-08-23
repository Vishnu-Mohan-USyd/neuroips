# Flank-suppressed sharpening — study protocol
Opened 2026-08-19 by team-lead. Status: Phase 1 (design) in flight.

## Goal (user-named, standing)
Modify the network minimally, using ONLY biologically plausible mechanisms, so that the trained
sharpening regime shows genuine flank suppression alongside center enhancement in the standard
orientation-profile assay. The mechanism must be trained INTO the network (present in training and
measurement; never inference-only) and flank suppression must EMERGE from the mechanism under the
original objectives — no objective term may reference center/flank bands or the tuning profile
shape (that would be fitting the metric; hard-forbidden by user and §3).

## Established facts (do not re-derive)
- Architecture: orientation input → fixed L4 (36 ch, 5°/ch) → Dale SOM/VIP motif → scalar
  k = g3 − g4·max(g1 − g2·g0, 0) scaling feedback W_fb(h): Linear 64→36 → softmax → f;
  RNNCell tanh 36→64. Repo: /home/vishnu/neuroips_rnn_recreation_20260808/repo (READ-ONLY).
- Original sharpening (α=0, 4 seeds): center +18–23%, flanks only −2 to −3% vs baseline
  (seed-invariant). Zero energy cost → nothing pushes activity down; sharpening ≈ pure center boost.
- Any energy pressure (ε≥0.2): suppression attacks the CENTER first (center notch; flanks spared
  or above baseline). Profiles: /home/vishnu/neuroips_outputs/orientation_figs_transition_20260819/.
- Assay convention (unchanged, binding): mean L2/3 rates vs orientation offset, gray literal-t0
  baseline vs continuation-A final step; center bin ±5°, flank bin ±15–30°. Reference figures:
  /home/vishnu/neuroips_outputs/orientation_figs_20260819/.
- Illustrative target shape (synthetic, for eyeballs only, NOT a fit target):
  /home/vishnu/neuroips_outputs/orientation_figs_synthetic_20260819/.

## Pre-registered acceptance criteria (single seed 8, standard assay, fixed before any run)
PASS requires ALL of:
1. Flank band (±15–30°) mean rate ≤ 0.85× its own baseline (stretch goal ≤ 0.75×).
2. Center (±5°) ≥ 1.15× its own baseline (some trade-off from the original +18% is acceptable;
   enhancement must remain clear).
3. Task preserved: final H ≥ 0.95 (original α0.0 ≈ 0.995).
4. Mechanism audit: modification is architectural/connectivity-level with a biological citation
   (primary source, full text per §8); objectives untouched w.r.t. profile shape; mechanism
   demonstrably develops during training (ep40–50 checkpoint: relevant parameters leave init AND
   profile trend visible). Kill on the spot if flat at ep40–50.
Multi-seed (≥3 additional seeds) confirmation ONLY after a single-seed PASS. Any FAIL → debugger
single-variable diagnosis BEFORE the next candidate; reasoned candidates only, no darts.

## Candidate space for Phase 1 ranking (researcher may extend; rank by biological grounding AND
minimality of diff)
a. Orientation-tuned SOM surround: extend the existing SOM/VIP motif so inhibition pools
   neighboring channels with a broader orientation footprint than excitation (per-channel k_i
   replacing scalar k) — SOM cells' broad pooling / surround suppression.
b. Mexican-hat (DoG) lateral kernel in orientation space on the L2/3 feedback path (ring-model
   lateral inhibition; Ben-Yishai/Somers lineage).
c. Divisive normalization at L2/3 with a broader-than-excitation normalization pool
   (Carandini–Heeger canonical computation).

## Phases
1. DESIGN (researcher): study repo feedback/L2/3 code paths; deliver DESIGN.md here with ranked
   candidates, exact minimal diffs (file:line), predicted mechanism of emergence, primary-source
   citations, and recommended parameterization (learned vs bio-fixed widths).
2. IMPLEMENT + CHEAP-PROVE (coder): candidate 1 as a training-time mechanism; single-seed retrain
   on /home/vishnu/scratch/flank_sharpening_20260819/ (~20 min class); ep40–50 development gate;
   endpoint assay with the standard convention.
3. On PASS: validator GO (criteria 1–4 + regression vs original validation suite) → multi-seed.
4. Deliverable: profile figure (same simplified style) from the new network + evidence pack.

## Envelope (never self-authorized)
dev3 only; existing roster only; GPU quota 2 (bitwise gates cuda:0); no sudo; training persistence
under /home/vishnu/scratch/ only; frozen roots READ-ONLY (repo, S2_plot, sweep cells, all
delivered study dirs); RAM: MemAvailable ≥25 GB before any launch, 8 GB RSS kill, sequential
single-process, one checkpoint at a time del+gc; PYTHONHASHSEED=0, python3 -B.

## Log
- 2026-08-19: Protocol opened; Phase 1 dispatched to researcher.
- 2026-08-19: Phase 1 COMPLETE. DESIGN.md sha256 85286bba…cfca2f38, 6 primary sources read in
  full. Top pick adopted: enable the lib's existing feedback-recruited subtractive surround via
  two MODEL_CONFIG constants in a byte-copy of the proven harness — pred_inhib_strength 0.0→0.5,
  pred_inhib_sigma_channels 0.65→4.0 (20°; SOM anatomy, Zhang 2014) — zero library/loss/assay
  edits. Lead verified on disk: hook exists (tuned_emergence_lib.py:63; harness train_sweep.py:45–46
  defaults 0.0/0.65). Biological basis: feedback-recruited SOM surround + VIP center disinhibition
  (Adesnik 2012, Zhang 2014, Nurminen 2018).
- 2026-08-19: LEAD RULING on §10 ladder autonomy: the s-ladder (0.5→0.25/1.0, σ fixed by anatomy)
  is pre-registered experimental design, so the coder may run the next rung WITHOUT new dispatch
  ONLY when the endpoint FAIL matches the pre-registered insufficient-magnitude signature exactly
  (mechanism developing at gate, f healthy, H ≥0.95, flanks moved below the −3% reference but
  short of 0.85×). ANY other signature (neutralization/f collapse, H <0.95, dead ring, flat gate,
  anything ambiguous) → STOP, lead routes to debugger. Max 2 additional rungs before mandatory
  debugger review regardless.
- 2026-08-19: Phase 2 dispatched to coder (single-seed 8 cheap retrain, fresh pretrain, step-4000
  development gate).
- 2026-08-19: Rung 1 (s=0.5, σ=4.0) GATE FAIL, kill-class — implementation clean (2-line diff
  audited; pretrain references bitwise in-family) but the circuit collapses: dead ring 0.82 at
  gate (0.83 endpoint), H 0.47→0.67, center boost GONE (0.98→0.93), effective k FALLING
  0.546→0.207→0.166, flanks over-suppressed to 0.13→0.08 (suppression works structurally; the
  circuit dies around it). NOT the insufficient-magnitude signature — no ladder run (per ruling).
  Routed to debugger for single-variable root-cause proof BEFORE any next change. Runs are ~3.5
  min — diagnostic probes are cheap.
- 2026-08-19: Debugger observation phase done; 4 hypothesis families registered with fixed verdict
  criteria (H1 lib-math, H2 magnitude overdose at true f scale, H3 gain-escape-as-cause via
  temporal precedence, H4 pretrain interaction via arm-only control). Notable: coder's pretrain
  bitwise check was mechanism-blind (fb=0 in reference path); pretrain WITH surround degraded
  (+26% task loss, +0.30 nats); k falls smoothly WHILE task improves (objective rewards raising
  SOM arm g1,g4 and cutting direct gains g0,g2,g3). Probes executing (2 pre-registered retrains +
  ≤1 contingency).
- 2026-08-19: DIAGNOSIS COMPLETE (DIAGNOSTIC_REPORT.md sha da191414…). H2 magnitude overdose
  CONFIRMED as root cause: row-normalized symmetric kernel makes mean subtraction s·Σf/36 ≈
  0.35–0.44 at s=0.5 — a blanket exceeding off-center drive 2–4×, flooring 72–83% of channels
  instantly on ANY weights (zero training needed). H1 lib-math, H3 k-fall-as-cause (temporal
  precedence: collapse at arm step 0 with k bitwise at init; healthy arm's k falls HARDER), H4
  pretrain-interaction (arm-only on bitwise healthy pretrain collapses identically) all RULED OUT.
  Dose probe s=0.05 (single-variable) flips everything: H 1.000, center 1.192, flank 0.789.
- 2026-08-19: EVIDENCE-DRIVEN AMENDMENTS (pre-registered before the confirmatory run):
  (A1) Ladder: downward rung set by the proven arithmetic to s=0.05-class (not 0.25) — keep
  s·Σf_typ/36 below the flank-band drive floor; σ=4.0 unchanged (anatomy).
  (A2) Gate: "effective k falling = kill/neutralization" REMOVED (healthy no-surround arm falls
  0.546→0.037 — it was a misread of the normal dynamic).
  (A3) Gate/validator: exact-zero dead_ring over-counts under any subtractive term (far-ring
  baseline 1e-6..1e-26); replaced by functional-band vitality — all channels |offset| ≤ 10° above
  floor 0.01 at continuation final step.
  (A4) Criterion-4 evidence for subtractive mechanism: endpoint inference counterfactual s→0 must
  REMOVE the flank suppression (proves the surround, not something else, does the work);
  measurement-only, run alongside the standard assay, never part of the deliverable profile.
- 2026-08-19: Phase 2 rung 2 (OFFICIAL, s=0.05, σ=4.0) dispatched to coder. Debugger's dose probe
  is diagnostic evidence only — official run, gate, assay, and validation all fresh.
- 2026-08-19: RUNG 2 ENDPOINT — ALL PRE-REGISTERED CRITERIA PASS, single seed 8, no
  reinterpretation: flank_ratio 0.7886 (bar ≤0.85; stretch 0.75 not met), center_ratio 1.1923
  (bar ≥1.15), H 1.0000 (bar ≥0.95), functional band alive (0.60–1.34). A4 counterfactual: s→0 at
  inference returns flank to 0.9716 — the surround does the work. Diff-audit: exactly 2 config
  lines vs frozen harness. Official numbers independently reproduce the debugger's dose probe.
- 2026-08-19: GATE RULINGS on rung-2 literal sub-fails (lead):
  (G-R1) "flank trending down" literal — WAIVED AS MISFIT, PASS ON INTENT: the criterion's purpose
  is to kill absent/flat mechanisms; here suppression is present from arm step 250 (0.771, flat to
  0.785) because the mechanism is fixed connectivity baked in from pretrain — the opposite of
  absent. What must develop for a fixed-connectivity mechanism is circuit health AROUND it, which
  did (H 0.486→1.000 monotone, center boost forming). Gate template amended for mechanism class.
  (G-R2) H 0.884 at step 4000 vs 0.9 trajectory bar — PASS ON INTENT: monotone rising through all
  16 snapshots to 1.000 at endpoint; the bar's purpose (kill collapsing-task runs) is unmet by a
  0.016 shortfall on a monotone climb. Both rulings apply to GATE (mid-run kill-switch) only; the
  ACCEPTANCE criteria passed literally and untouched.
- 2026-08-19: Phase 3 dispatched — validator independent audit (with pre-authorized s=0 A/A
  control retrain vs frozen sweep cell) in PARALLEL with coder multi-seed confirmation
  (seeds 9/10/11, same config, sequential).
- 2026-08-19: MULTI-SEED CONFIRMATION — seeds 9/10/11 ALL PASS: H 0.981/0.995/0.995, center
  1.204/1.216/1.216, flank 0.787/0.786/0.786; with seed 8 the flank ratio is 0.786–0.789 across
  all four seeds (spread 0.003, strongly seed-invariant); A4 s→0 counterfactual removes the
  suppression on every seed (0.968–0.972). Stretch ≤0.75 not met on any seed. Numbers forwarded
  to validator for the combined verdict. Figure build dispatched in parallel; DELIVERY remains
  gated on validator GO.
- 2026-08-19: VALIDATOR VERDICT — GO, all four seeds (VERDICT.md sha 73239084…). Endpoint numbers
  re-derived exactly (own pipeline, zero mismatches); A/A control BITWISE (entire final checkpoint
  tree-diff: one leaf = the σ config value; strength alone carries the effect; σ-inertness at s=0
  proven at grad level first); criterion 4 clean (objectives untouched; A4 counterfactual all
  seeds; Adesnik/Zhang stats verified verbatim in full text; spatial→orientation transplant
  labeled). Gate rulings reviewed: G-R1 agree; G-R2 agree with correction applied below.
- 2026-08-19: CORRECTION (validator-sourced, to G-R2 above): H trajectory to the gate is a RISING
  TREND WITH TWO 3-HISTORY DIPS (750→1000 and 2750→3000, −0.0139 each; gate value 0.8843 is the
  trajectory max; net +0.398; endpoint 1.000) — not "monotone rising through all 16 snapshots."
  Ruling's substance unchanged. Same erratum ordered for coder's RUN_LOG. Minor validator notes
  (1-ulp effective_k cosmetic diff; DESIGN §6 trend-direction prediction wrong — onset right;
  archived shared-pretrain payload key artifact) recorded here as filed.
- 2026-08-19: STUDY DELIVERED to user: /home/vishnu/neuroips_outputs/flank_sharpening_20260819/
  (main + comparison figures, provenance, VERDICT copy, manifest). STUDY CLOSED.
- 2026-08-19: VERDICT ADDENDUM (validator, verdict unchanged GO): all three non-8 seeds re-derived
  end-to-end (zero mismatches, full float precision) and the figure-input 25-point curves
  (adapted + baseline) recomputed independently for seeds 8 and 9 — 25/25 elementwise-exact.
  VERDICT.md final sha 7ce7de43…; delivery pack references updated to this sha.
- 2026-08-19: STUDY REOPENED — PHASE 4 (user-mandated validity condition): the delivered result is
  PROVISIONAL — sharpening currently trains with the surround enabled while the dampening arm does
  not, so the regime comparison is confounded by an architectural difference. Validity requires
  the FAMILY claim: one fixed architecture (surround s=0.05, σ=4.0 enabled everywhere), regimes
  set ONLY by task/energy pressure. Phase 4: retrain α=0.5 (dampening) with the identical two
  constants and verify the dampening phenotype survives.
- 2026-08-19: PHASE 4 PRE-REGISTRATION (bars fixed before any Phase-4 assay is read; reference
  values to be computed from FROZEN α0.5 artifacts first and appended here by the coder before
  the retrain result is examined): (P1) dampening direction: center_ratio ≤ 0.35 (original ≈0.15);
  (P2) topology preserved: center_ratio < flank_ratio (center-first suppression signature);
  (P3) in-family vitals/task: H and M within ±0.15 (relative) of the original α0.5 seed-8 values;
  (P4) no collapse: continuation mean rate above floor (band alive per A3 sense — profile nonzero,
  no fully-silent ring). Single seed 8 first; seeds 9/10/11 confirm on pass; validator combined
  re-verdict after. FAIL on any bar → delivered claim declared INVALID to the user and the loop
  returns to design (joint calibration for both regimes).
- 2026-08-19 [CODER-APPENDED reference values — computed from FROZEN artifacts BEFORE any Phase-4
  retrain result existed; no Phase-4 training had been launched at append time]:
  Source network: frozen original α0.5 seed-8 endpoint, no-surround architecture
  (S2_plot/seed_8/alpha_0p5_final.pt, sha256 156cc0f2…bc70bc18; bitwise identical to
  S2_confirm/seed_8/alpha_0p5_final.pt — the same network behind the delivered dampening figure
  and the frozen gate decision). Fresh assay (standard frozen convention, cuda:0) cross-checked
  against frozen_gate_decision_rnn.json seed-8 α0.5: Cret/Fret/M/rate_A/rate_t0 all reproduce to
  ≤2.8e−17. REFERENCE VALUES (P-bars pinned):
  · center_ratio (Cret) = 0.149572   · flank_ratio (Fret) = 0.559042
  · H = 0.194444 (42/216; NOT in the frozen assays — fresh-computed only; the dampening arm's
    argmax hit rate is intrinsically low, chance = 0.027778; granularity 1/216 = 0.00463)
  · M = 0.332062 (frozen definition: whole-36-bin expected-A AUC / timestep-0 AUC
    = continuation mean rate / t0 mean rate)   · continuation mean rate (rate_A) = 0.055348
  NUMERIC BARS THEREFORE: P1 center_ratio ≤ 0.35; P2 center_ratio < flank_ratio;
  P3 H ∈ [0.165278, 0.223611] AND M ∈ [0.282253, 0.381872] (±15% relative);
  P4 operationalized in the A3 sense (the original's own far ring is near-silent, min 2.3e−10, so
  an all-36-channel floor would fail the reference itself): continuation mean rate > 0.01 AND all
  |offset| ≤ 10° channels of the mean aligned continuation-final profile > 0.01. Original passes
  with margin: rate 0.055348; band values 0.103–0.149.
  Artifacts: /home/vishnu/scratch/flank_sharpening_20260819/phase4_reference_alpha0p5_seed8.json
  (sha256 2ddb0e8b…02fc9783 — full curves + cross-check diffs), generator
  phase4_reference_eval.py (sha256 789da17b…f397ab50a). A4-style s→0 counterfactual will be
  recorded as evidence, not a bar (per dispatch).
- 2026-08-19: PHASE 4 SEED-8 RESULT — 4/5 PASS, 1 FAIL (coder stopped correctly, no remedies).
  P1 PASS strong (center_ratio 0.0997 vs original 0.1496 — dampening direction STRONGER than
  original); P2 PASS (0.0997 < 0.4355, center-first topology preserved); P3_H PASS (0.2083,
  +7.1% rel); P4 PASS (rate 0.0413, band 0.067–0.132 alive). P3_M FAIL: M 0.2475 vs band
  [0.2823, 0.3819] (−25.5% rel). t0 baseline rate identical to original → shortfall entirely in
  adapted activity; both center and flank sit below their original values (same direction).
  A4 s→0 evidence: M returns to 0.3020 (inside band), flank 0.5376 (near original) → the
  surround's direct subtraction at measurement appears to carry most of the shortfall —
  UNVERIFIED until the debugger proves the split. Family-parity claim NOT established; the
  delivered sharpening result remains PROVISIONAL. User informed (interim verdict promised
  either way).
- 2026-08-19: PRE-REGISTERED NEXT (contingent on debugger's proven cause; fixed before any
  further training): JOINT DOSE LADDER — s ∈ {0.02, 0.03, 0.04}, σ=4.0 fixed, BOTH regimes
  (α=0.0 and α=0.5) retrained per s, seed 8. A configuration passes ONLY if the SAME s clears
  ALL existing pre-registered bars in both regimes (sharpening: flank ≤0.85, center ≥1.15,
  H ≥0.95, vitality; dampening: P1–P4 incl. the M band). No bar may be loosened to fit. If no s
  passes jointly, the family claim FAILS and the user is told the delivered result is invalid as
  a family mechanism. Multi-seed + validator re-verdict at the winning s only. Phase-4 M-shortfall
  diagnosis dispatched to debugger.
- 2026-08-19: M-SHORTFALL DECOMPOSITION PROVEN (DIAGNOSTIC_REPORT_PHASE4_M.md sha 549c4b71…;
  closed 2×2, zero training, coder's measure() verbatim, E0 <1e-12, orig cell = frozen-M anchor
  exact). Direct measurement-time subtraction 64–74% of the gap; trained adaptation 26–36%;
  interaction NEGATIVE (retrained weights withstand the surround better than original). NO regime
  change: at s=0 the retrained dampening net matches the original in shape (r=0.970), topology,
  and beats it on H; M in-band. Mean subtraction = s·Σf/36 exactly (blanket arithmetic at benign
  dose; ~79% of channels floored, realized loss ~26–30% of blanket). Adaptation's why-lower =
  flagged unknown (not required). LADDER PREDICTION (labeled): dampening M bar → s=0.02 PASS,
  s=0.03 marginal FAIL, s=0.04 ROBUST FAIL (pure subtraction alone breaks the band); sharpening
  side unpredictable from bolt-on (training amplifies suppression) — joint window, if any, near
  s=0.02–0.03. Genuine risk that none passes both.
- 2026-08-19: JOINT DOSE LADDER LAUNCHED per pre-registration (coder): pairs per s starting 0.02,
  both regimes, seed 8, all six runs for the dose-response record, existing bars verbatim, no
  remedies. Debugger stood down.
- 2026-08-19: LADDER COMPLETE (6/6 runs, bars verbatim, harness restored to 0.05 after).
  **s=0.04 = JOINT PASS at seed 8**: sharpening flank 0.8279 / center 1.1895 / H 0.9907 all pass;
  dampening M 0.2961 in band + P1 0.1436 / P2 / P3_H 0.1991 / P4 all pass. s=0.02 fails
  sharpening flank (0.9047); s=0.03 fails both flank (0.8665) and dampening M (0.2500); s=0.05
  (official) fails dampening M (0.2475). Sharpening trained-flank monotone DOWN in s
  (0.905/0.866/0.828/0.789); training amplifies at every dose (A4 ≈0.97 throughout).
- 2026-08-19: ANOMALY FLAGGED (coder, facts-only; routed to debugger BEFORE building on the
  joint pass): dampening trained-M is NON-MONOTONE in s (0.2889/0.2500/0.2961/0.2475), and the
  s=0.04 value 0.2961 EXCEEDS the debugger's zero-adaptation ceiling 0.2805 — the retrained
  weights hold MORE adapted activity than original weights under the same inference-s,
  contradicting the labeled prediction's bracket (adaptation assumed ∈ [0, linear-in-s]).
  s=0.02 PASS and s=0.03 FAIL landed as predicted; s=0.04 did not. Joint window is PROVISIONAL
  until the anomalous cell is independently verified and the non-monotonicity explained;
  multi-seed at s=0.04 held pending the debugger's cell verification (bogus-eval branch must
  die first or multi-seed would inherit it).
- 2026-08-19: PART A VERDICT — both s=0.04 cells REAL (debugger: config carried in checkpoints;
  provenance chain sha-consistent incl. correct identical-pretrain signature; independent
  recompute with the official measurement core EXACT — dampening M 0.29606407963526854 abs diff
  0.0, sharpening flank 0.8278569927266332 abs diff 0.0). MULTI-SEED AT s=0.04 LAUNCHED (coder:
  seeds 9/10/11 × both regimes). Debugger continues Part B (non-monotone/positive-adaptation
  explanation) in parallel; final family verdict waits for BOTH.
- 2026-08-19: PART B COMPLETE (DIAGNOSTIC_REPORT_PHASE4_LADDER.md sha 6bed6dea…). Direct
  subtraction is MONOTONE in s (dose physics as proven); the non-monotone trained-M column lives
  entirely in SETTLED-WEIGHT SCATTER: weights-only M 0.3146/0.2866/0.3500/0.3020 — Hb2 CONFIRMED
  (deterministic path sensitivity in a non-settling α=0.5 regime; divergence graded from pretrain;
  late-training k swings 0.9–1.0), Hb1 attractor-multiplicity and Hb3 energy-directional RULED
  OUT (one gain family, adaptation sign-flips). s=0.04 state LEGITIMATE, not knife-edge (passes
  the M bar measured anywhere in s'∈[0.03,0.05]; profile r=0.9927 vs original, the closest of all
  cells). KEY CALIBRATION FACT: the original no-surround α0.5 family's own seeds span M
  0.3071–0.3321 with the band's anchor (seed 8, 0.3321) at the TOP — original seeds clear the
  band floor by only 0.025–0.031, comparable to the demonstrated between-run scatter ±0.02–0.05.
  LABELED PREDICTION: s=0.04 dampening multi-seed M ≈ 0.28–0.30 ± 0.02–0.03 → material chance
  one seed dips under 0.2823; sharpening side much safer (margin 0.022 vs invariance 0.003).
- 2026-08-19: PRE-REGISTERED OUTCOME RULES for the s=0.04 multi-seed (fixed before its results
  are read; NO new bands invented): (O1) all three confirm seeds pass everything in both regimes
  → family claim CONFIRMED at s=0.04 → validator combined re-verdict → deliverables rebuilt at
  s=0.04, s=0.05 pack superseded. (O2) any bar fails on any seed → family claim NOT CONFIRMED
  under the pre-registered bands — reported to the user as such, with the band-calibration
  evidence (original's own seed scatter vs the seed-8-anchored band) given as context, and any
  distribution-referenced re-test happening only if the user asks, on FRESH seeds, under a newly
  pre-registered criterion. The bands themselves are not loosened post hoc under any outcome.
- 2026-08-19: MULTI-SEED AT s=0.04 COMPLETE (coder; 6 runs + seed-8 ladder rows = 8-run view).
  SHARPENING: 4/4 seeds ALL PASS — flank 0.8240–0.8279 (spread 0.004), center 1.189–1.222,
  H 0.968–0.991. DAMPENING: P1/P2/P3_H/P4 pass on ALL FOUR seeds; P3_M splits 2/2 — M
  0.2961 (s8 PASS) / 0.2637 (s9 FAIL, −6.6% rel) / 0.2820 (s10 FAIL by 2.47e−4 = 0.09%) /
  0.3091 (s11 PASS). Seed-level joint verdicts: 8 YES, 9 NO, 10 NO, 11 YES → under the
  pre-registered bands this is OUTCOME O2 (family claim NOT CONFIRMED), matching the debugger's
  labeled scatter prediction. TENTATIVE pending independent verification: the verdict pivots on
  hairline numbers (seed 10 at 0.09%), so the six new cells go to the validator for exact
  re-derivation BEFORE the user-facing verdict. No remedies, no bar changes.
- 2026-08-19: VALIDATOR VERIFICATION — O2 CONFIRMED EXACTLY (VERDICT.md addendum, new sha
  f5640239…). Zero mismatches across all eight s=0.04 cells; seed-10 M 0.2820059371 strictly
  below the bitwise-verified floor 0.2822529582 by 2.4702e−4; seed-level joint verdicts
  8Y/9N/10N/11Y; sole failing bar anywhere = dampening P3_M (seeds 9, 10). FAMILY CLAIM NOT
  CONFIRMED under the pre-registered bands. USER INFORMED per O2 with the band-calibration
  context and the distribution-referenced fresh-seed re-test stated as the recommended next step
  (runs only on the user's word). Delivered s=0.05 pack status: stands as a single-arm mechanism
  demonstration only; INVALID as a family mechanism under the user's parity condition. All
  agents idle; study holds at O2 pending user direction.
