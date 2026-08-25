# DESIGN.md — Weight-set strategy investigation on the surround architecture (Phase 1)

Researcher, 2026-08-23. Governing protocol: `PROTOCOL.md` in this directory. Read in full for
this design: `transplant_20260818/PROTOCOL.md` (308 lines, incl. AMENDMENT-1) and `REPORT.md`
(375 lines); `weight_strategy_20260811/REPORT.md` (WS, the transplant study's predecessor);
`flank_sharpening_20260819/PROTOCOL.md` (log incl. Phase-4 pre-registration, ladder, multi-seed,
O2) and `VERDICT.md` (incl. Addendum 2, the validator's exact re-derivation of all eight s=0.04
cells); donor run dirs and `training_summary.json` files under
`/home/vishnu/scratch/flank_sharpening_20260819/runs/ladder_s0p04/`. This document is
design-only: no code, no runs. Everything below is fixed BEFORE any measurement.

Notation: component letters (CELL, FB, GAINS) as in the original study; cell ID = (CELL, FB,
GAINS) with P = the seed's s=0.04 pretrain tensor, T = the regime's s=0.04 trained-arm tensor.
Dispatch alias map: C=TPP, F=PTP, G=PPT, CF=TTP, CG=TPT, FG=PTT, CFG=TTT, host-only=PPP.

---

## 1. ORIGINAL CONVENTIONS — extracted verbatim, with inherit/adapt rulings

### 1.1 Host convention (fresh-init vs pretrain host, init seeding)
Original (transplant_20260818 §2, construction rule): hybrids are built by taking the **body
from the arm checkpoint** and overwriting the P-designated components with **exact float32
tensor copies from the seed's `common_pretrain_final.pt` — no arithmetic**. Equivalently: the
host is the seed's PRETRAIN network (task-pretrained 3000 steps, never exposed to the α arm
objective), never a fresh random init. PPP = the pretrain state itself; AMENDMENT-1: because the
frozen evaluator validates `checkpoint["alpha"]`, the PPP checkpoint is written **under both arm
filenames with own-arm alpha metadata** (REPORT §1: with one shared byte-stream,
`whole_profile_retention(dir, 0.5)` raises `ValueError: … wrong alpha metadata`; with per-arm
metadata M(α0.0)=M(α0.5)=1.4176421670673902 bit-identical; `alpha` is the only non-state_dict
field that differs between arm checkpoints).

**Ruling — INHERIT with one adaptation.** The new protocol's Goal says "untrained fresh host"
(user's phrasing); the Design-constraints clause mandates inheritance from transplant_20260818.
Resolution: host = the seed's **s=0.04 `common_pretrain_final.pt`** ("untrained" = not
arm-trained; it IS task-pretrained). Reasons: (a) inheritance is mandated and the fresh-random
host has no precedent in any frozen study; (b) a random-init host lacks the pretrained
representation every component co-adapted to — all 14 non-PPP cells would fail trivially and ρ
would be uninterpretable; (c) the T0 partition proof (§1.4) makes the pretrain host exactly the
all-P corner of the factorial. Adaptation: the host carries the surround config (s=0.04, σ=4.0)
from pretrain step 0 — the mechanism is therefore present in the host at measurement, as the new
protocol requires ("surround kernel config-carried; present in host by config"). Verified on
disk: both regime dirs of seed 8 record `common_pretrain_state_sha256`
`4c5b1a320300630cafcf1b2cbce77dd3c05abf7128aa3eb3eb24b46457bc4236` (identical), and the flank
validator verified all four seeds' within-seed pretrain state-bitwise identity (VERDICT.md
Addendum 2). AMENDMENT-1's dual-filename PPP construction is inherited verbatim.

### 1.2 Splicing mechanics
Original: state-dict splicing only. Components (T0-proven exhaustive): **CELL** =
`gru.weight_ih` (64×36), `gru.weight_hh` (64×64), `gru.bias_ih`, `gru.bias_hh`; **FB** =
`W_fb.weight` (36×64), `W_fb.bias` (36); **GAINS** = `circ_raw` (5). Exact float32 copies, no
arithmetic; every state_dict key traced to its declared source (construction audit, 56/56 cells
× 12 keys, bitwise); assay rebuilds each net via
`build_tuned_from_config(checkpoint["tuned_net_config"])` + `load_state_dict`. The 12-key
state_dict (verified on an s=0.04 donor, read-only): `W_fb.bias, W_fb.weight, circ_raw,
decoder_gain_raw, ff_weight, gru.bias_hh, gru.bias_ih, gru.weight_hh, gru.weight_ih,
local_comp_strength_raw, readout_cos, readout_sin`. The surround kernel (`pred_inhib_weight`)
is a `persistent=False` buffer — **never in the state_dict**, rebuilt from config at load
(verified by the flank validator) — so splicing never touches it and every hybrid inherits the
kernel from its checkpoint's `tuned_net_config`. **INHERIT unchanged**; hybrid checkpoints must
carry the s=0.04/σ=4.0 `tuned_net_config` (build gate, §3.6).

### 1.3 Combination matrix
Original: full 2³ factorial per regime — PPP, TPP, PTP, PPT, TTP, TPT, PTT, TTT — × 2 regimes ×
4 seeds = 64 table cells, 60 distinct nets (PPP shared per seed, computed once, entered in both
tables, verified bitwise-identical under both filenames). **INHERIT unchanged** (= the
dispatch's 7 combos + host-only).

### 1.4 T0 partition proof (exhaustiveness)
Original T0: for every arm×seed (8/8), the pretrain→arm state_dict diff touches **only** the 7
CELL/FB/GAINS tensors; all other keys bitwise identical. Δ-norm signature: α0.0 mass in
W_ih/W_fb (‖Δ‖≈21–23); α0.5 mass in W_hh/circ_raw. k by source: pretrain +0.5457; α0.0
+0.037..+0.048; α0.5 −3.47..−3.91. **INHERIT as gate G1** (re-proven at s=0.04 before any
build). Expected to hold: no new trainable parameter exists at s=0.04 (`pred_inhib_weight` is a
non-persistent buffer; `local_comp_strength_raw` frozen — donor summaries record
`local_comp_raw_byte_stable: true` with initial=final sha `df3f6198…`, all cells).

### 1.5 FB controls — provenance correction (load-bearing for this design)
The new protocol lists FB controls "(random / magnitude-matched / rotation-misaligned) …
inherited/adapted from transplant_20260818's frozen protocol." **Measured fact: no such controls
exist in transplant_20260818 (protocol or report), in WS, or anywhere in the analysis tree** — a
recursive grep for random/magnitude-matched/misaligned FB artifacts returns only the new
protocol itself. The original factorial is strictly P/T. The nearest precedents are: (a) the
original's FB=P cells (pretrain FB in trained context), and (b) WS A2's single-variable W_fb
edits — rescale (direction kept, norm reset) and reverse (norm kept, direction reset) — which
proved for α0.0 that the **direction** of the FB rewrite carries sharpening (norm fraction ≥0.5
in 1/4 seeds only; reverse edit costs more than norm reset in 3/4).

Consequently the baseline paragraph's two FB claims are today supported only in weakened form:
- "misaligned trained FB is WORSE than untrained" — supported as: every α0.0 cell that inserts
  trained FB without its co-trained partners lands at or below the FB=P counterpart on placement
  (PTP −0.33..−0.47 vs PPP 0; TTP −0.31..−0.37 vs TPP −0.03..−0.20, 4/4; PTT −0.55..−0.62 vs
  PPT −0.55..−0.58, ≈wash), plus WS's reverse-edit result. No literal rotation control was ever
  run.
- "untrained (random) FB suffices, 4/4 (k applies negative gain to a magnitude-preserving random
  direction)" — the measured fact is that **pretrain** FB suffices (α0.5 TPT: ρ_decode
  0.91–0.98, ρ_M 0.92–0.97, no CE trip, 4/4). Pretrain FB is not random — it is task-informative
  (PPP decode_A 0.742–0.758). A literal random or magnitude-matched-random FB was never tested;
  the "random direction" gloss is a plausible but unproven mechanism reading.

**Ruling: the three FB controls are NEW constructions, defined in §2.3 of this design and
pre-registered here.** They convert the baseline's overstatement into the registered hypotheses
H-C1/H-C2 (§4.5). This discrepancy is flagged to the lead in the Phase-1 report.

### 1.6 Effect metrics and thresholds (original, verbatim)
- **E1** M = `whole_profile_retention` (frozen evaluator; M = whole-36-bin expected-A AUC / t0
  AUC = continuation mean rate / t0 mean rate).
- **E2** final-step placement hit A_on_y (assay transition index t=3; the same quantity the
  flank study calls H at FINAL_STEP=4 — numerically confirmed: α0.5 seed-8 original = 0.1944 =
  42/216 in both studies' artifacts).
- **E3** decode_A_minus_B and B_minus_A_rate (+ decode_A, decode_B).
- Chimera gate: any cell with max mean CE_A > 3·ln36 = **10.7506** trips; a tripped cell cannot
  confirm a competence-dependent claim (house rule; that seed becomes UNRESOLVABLE for it).
- **ρ_m(X) = (m(X) − m(PPP)) / (m(TTT) − m(PPP))**, per seed. Denominator floors: decode 0.04,
  rate 0.008, hit 0.15, M 0.05; below floor ⇒ UNREADABLE, raw values reported.
- Bands: **carries** F = ρ ≥ 0.75; **partial** ∂ = 0.25 < ρ < 0.75; **absent** 0 = ρ ≤ 0.25.
- Claims evaluated per seed, 4/4 to confirm; a miss is reported as a miss; no post-hoc
  re-lettering; raw values reported alongside every ρ.
**INHERIT all**, with the α0.5 rate-coordinate demotion and the new profile coordinates of §3.

### 1.7 Headline result table (original study — the baseline this study tests for change)
α0.0 factorial, registered scalar ρ_hit (s8/s9/s10/s11):
| cell | ρ_hit | class |
|---|---|---|
| TPP (C) | −0.17/−0.18/−0.03/−0.20 | 0 |
| PTP (F) | −0.33/−0.40/−0.43/−0.47 | 0, below baseline |
| PPT (G) | −0.57/−0.58/−0.58/−0.55 | 0, below baseline |
| TTP (CF) | −0.35/−0.32/−0.37/−0.31 | 0, below baseline |
| TPT (CG) | **+0.46/+0.49/+0.42/+0.47** | ∂ — best partial |
| PTT (FG) | −0.55/−0.55/−0.60/−0.62 | 0, below baseline; sole α0.0 CE trip (s8, 11.311) |
| TTT | +1.00 | F |
Verdict language (REPORT §9): "on placement, CELL+GAINS recovers 42–49% and no single component
recovers anything." 5 of 6 partial cells sit BELOW the untrained baseline in all four seeds.
R1a/R1b/R1c (FB sufficiency/necessity/ordering) all MISSED — sharpening = co-adapted trio.

α0.5 factorial: registered scalar ρ_rate **UNREADABLE 4/4** (denominators −0.000123/−0.007102/
+0.001298/+0.000078 vs floor 0.008 — the untrained net already shows the arm's absolute B−A rate
difference). On the readable markers: **TPT** ρ_decode 0.91–0.98, ρ_M 0.92–0.97, no trip — the
carrier ("CELL+GAINS reaches 0.91–0.98 of the trained arm without FB and without losing
competence"); TTP (GAINS removed) overshoots M ABOVE the untrained baseline (1.659–1.681 > PPP
1.415–1.418) and collapses expected-A decoding (decode_A 0.031–0.068); PPT and PTT trip CE 4/4
(max CE_A up to 16.587); R4a M(TPT)−M(TTT) ≤ 0.10 PASS 4/4; R4b PASS 4/4. GAINS is the component
whose removal moves every α0.5 marker (TTT→TTP moves M by ~38× more than TTT→TPT).

---

## 2. NEW RUN MATRIX (exact)

### 2.1 Donors and hosts (verified on disk)
Root: `/home/vishnu/scratch/flank_sharpening_20260819/runs/ladder_s0p04/` — 8 run dirs
`alpha0p{0,5}_seed{8,9,10,11}/seed_N/`, each holding `common_pretrain_final.pt` +
`alpha_0p{0,5}_final.pt` (+ `_latest.pt`, `training.jsonl`, `training_summary.json`).
Provenance already validated end-to-end by the flank validator (VERDICT.md Addendum 2): s=0.04 /
σ=4.0 carried in run_start, pretrain config, and final config of every cell; step 8000;
alpha/task_weight 0.0/1.0 and 0.5/0.5; within-seed pretrain state-bitwise identical across
regimes; all 8 final states distinct; freeze_local_comp bytes-stable. Seed-8 k (from
training_summary, to be re-derived as G5): α0.0 **+0.04727** (g = 0.5496/0.8699/0.5496/0.5502/
0.8856), α0.5 **−3.5016** (g = 0.2674/1.9048/0.2674/0.2776/2.0614) — same sign structure as the
no-surround arms (+0.037..+0.048 / −3.47..−3.91).

Donor caveat (recorded, does not block): under the flank study's pre-registered bands the s=0.04
family verdict is **O2** — dampening P3_M fails on seed 9 (M 0.2637, −6.6% rel) and seed 10
(0.2820, below floor 0.28225 by 2.47e−4); sharpening passes 4/4. All eight states are
LEGITIMATE, scatter-explained (debugger Part B), and every transplant reading here is
within-seed ρ (own TTT/PPP denominators), so the strategy map remains well-defined; but any
dampening conclusion generalizes to "the s=0.04 α0.5 solution family," not "the in-band
phenotype family." Seeds 9/10 dampening rows carry this caveat in the report.

### 2.2 Core factorial
8 cells (PPP…TTT) × 2 regimes × 4 seeds = **64 table cells, 60 distinct nets** (PPP shared per
seed via AMENDMENT-1 dual-filename). Identical to the original matrix, donors swapped to s=0.04.

### 2.3 FB control cells (new constructions; RNG fully pinned)
Control FB variants, replacing BOTH `W_fb.weight` and `W_fb.bias` unless stated:
- **R (random-init):** instantiate a fresh net via `build_tuned_from_config(host config)` under
  `torch.manual_seed(20260823)` on CPU; take its `W_fb.weight`/`W_fb.bias`. Lib-faithful init
  (nn.Linear default), regime-independent. Note: init Frobenius norm ≈ half the pretrain W_fb
  norm (WS rescale record: σ_pre 7.11–8.50, σ_arm(α0.0) 13.24–13.92) — R therefore also probes
  low FB "softmax temperature."
- **N (norm-matched random):** the R tensors rescaled so ‖W_N‖_F = ‖W_fb^T(regime,seed)‖_F and
  ‖b_N‖₂ = ‖b^T‖₂ (exact scalar multiply; relative error gate <1e−6). Random direction at
  trained magnitude — the literal test of "magnitude-preserving random direction."
- **Q (rotation-misaligned trained):** W_Q = W_fb^T @ Q with Q a Haar-random orthogonal 64×64
  (QR of a `torch.manual_seed(20260824)` Gaussian, sign-fixed diag(R)>0); b_Q = b^T unchanged
  (bias lives in channel space, untouched by an h-space rotation). Preserves Frobenius norm and
  every row 2-norm; destroys h-space alignment. Gates: ‖QᵀQ−I‖_∞ < 1e−5,
  |‖W_Q‖_F/‖W_fb^T‖_F − 1| < 1e−5.
One draw of R and one Q are used for ALL cells (fixed seeds above); per-regime/seed N and Q
derive from that regime×seed's trained FB.

Contexts:
- **T·ctrl·T** (CELL=T, GAINS=T, FB=control) — the adjudicating context: 3 controls × 2 regimes
  × 4 seeds = **24 cells**.
- **P·ctrl·P** (control FB alone in the host) — calibration vs PTP, seed 8 only: R (regime-
  independent, entered in both factorial tables) + N, Q per regime = 5 nets, **6 table cells**.
Total with core: **94 table cells, 89 distinct nets**.

### 2.4 s→0 inference counterfactual on hybrids (measurement-only; inherited from flank A4)
Re-assay selected cells with `pred_inhib_strength=0` in the rebuilt config (σ inert at s=0 —
proven end-to-end by the flank validator), everything else identical. Registered list: {TTT,
TPT, PPT, PTP, PPP} × 2 regimes × seed 8 (10 re-assays), plus every cell whose seed-8 ρ_flank ≥
0.25 (bounded ≤8 more), with 9/10/11 confirmation only for cells cited in a load-bearing claim.
Purpose: attribute any transplanted flank suppression to the surround path (Δflank = flank(s) −
flank(0)) rather than to CELL/GAINS side-effects.

### 2.5 Load-bearing cells (named now)
Sharpening: **TPT** (does CELL+GAINS still recover ~0.4–0.5 and nothing more?), **TTP** (the
kernel-does-the-shaping test: does trained CELL+FB now carry flank without GAINS?), **TQT vs
TPT** (alignment premium), **PTP** (FB-alone flank movement — novel physics), TTT/PPP anchors.
Dampening: **TPT** (does CELL+GAINS still carry?), **TNT and TRT** (the literal
random-direction sufficiency test), **TTP** (overshoot repeat), **PPT/PTT** (CE-trip repeat).
These are the cells whose readings the validator must independently re-derive (Phase 3).

---

## 3. PRE-REGISTERED READING RULES (fixed before any measurement)

### 3.1 Endpoints per cell
Every cell gets: E1 M (frozen evaluator, dual-filename dirs); E2 final hit A_on_y; E3 decode
markers + CE profile; **and the flank-study profile coordinates** (registered conventions:
aligned mean profile, PLOT_OFFSETS ±60°, CENTER_OFFSETS ±5°, FLANK_OFFSETS ±15–30°,
FINAL_STEP=4, adapted = continuation-A final step, baseline = literal t0):
**center_ratio** = mean over CENTER_OFFSETS of adapted/baseline; **flank_ratio** = mean over
FLANK_OFFSETS; vitality per P4 sense (continuation mean rate > 0.01 AND all |offset|≤10°
channels > 0.01).

### 3.2 Primary effect metrics per regime (dispatch-fixed)
- **Sharpening (α0.0): ρ_center and ρ_flank** (the phenotype the surround was built for),
  with **ρ_hit as the competence/placement primary** (it was the original's registered scalar
  and stays the sharpest coordinate). A component set "carries sharpening" only if it carries
  BOTH the profile phenotype (ρ_center AND ρ_flank ≥ 0.75) and placement (ρ_hit ≥ 0.75), 4/4
  readable seeds, no CE trip. Partial/absent read per coordinate; dissociations (e.g. flank
  carries while hit doesn't) are reported as dissociations — that outcome is itself a registered
  question (§4.4 Q1).
- **Dampening (α0.5): ρ_M and ρ_center** (center_ratio is the direction-defining P1 metric and
  had no analog in the original grid; M was the original's strongest readable marker).
  "Carries" = both ≥ 0.75, 4/4 readable, no trip. **ρ_rate is DEMOTED to raw-report-only** —
  adaptation of the original registration in light of its own measured failure (S1/EC2:
  UNREADABLE 4/4; the pretrain baseline reproduces the arm's absolute B−A difference). Raw rate
  and ρ_decode are companions.

### 3.3 ρ machinery and floors
ρ_m(X) = (m(X)−m(PPP))/(m(TTT)−m(PPP)) per seed per factorial. Inherited floors: hit 0.15,
decode 0.04, rate 0.008, M 0.05. New floors (same |TTT−PPP| ≥ floor form): **center_ratio 0.05,
flank_ratio 0.05**. Below floor ⇒ UNREADABLE, raw values reported, no adjudication on that
coordinate. Labeled expectation (not a bar): the sharpening **flank** denominator is the one at
genuine risk — the host itself has the surround active (pretrain f is informative, PPP decode_A
was 0.74 in the original), so host flank_ratio may sit anywhere in ≈0.85–0.97 against TTT
0.824–0.828; if it floors out, the flank question is answered descriptively (raw flank + the s→0
counterfactual) and the sharpening verdict rides on ρ_center + ρ_hit. Dampening denominators
are expected comfortably readable (TTT center 0.093–0.164 vs host ≈1; M far below host).

### 3.4 Verdict bands
Inherited verbatim: carries F = ρ ≥ 0.75; partial ∂ = 0.25–0.75; absent 0 = ρ ≤ 0.25 (values
below 0, i.e. below the untrained baseline, are reported as "0, below baseline"). Confirmation:
seed 8 primary; a claim becomes part of the strategy map only at 4/4 readable-seed agreement
(house rule: a CE-tripped cell makes that seed UNRESOLVABLE for competence-dependent claims;
verdict then rests on the untripped seeds and says so).

### 3.5 Chimera gate
Inherited verbatim: trip = max mean CE_A > 3·ln36 = 10.7506 on the A stream; trips reported per
cell; house rule as §1.6.

### 3.6 Gate chain (Phase 2 must pass, in order, before any cell is read)
- **G6 (new):** MANIFEST.sha256 of all 8 donor dirs' files written BEFORE any build; re-verified
  after all measurement (scratch is not a frozen root; this substitutes for the original G1b
  md5-vs-frozen-list gate). Seed-8 pretrain sha must equal the recorded
  `4c5b1a320300630cafcf1b2cbce77dd3c05abf7128aa3eb3eb24b46457bc4236`.
- **G1 (T0-analog):** partition proof per regime×seed (8/8): pretrain→arm diff confined to the 7
  CELL/FB/GAINS tensors, all other keys bitwise (incl. `local_comp_strength_raw`, `ff_weight`,
  readout/decoder tensors). Δ-norm table recorded (input to §4.1).
- **G2 (pretrain equality):** per seed, the two regime dirs' `common_pretrain_final.pt`
  state_dicts bitwise identical.
- **G0 (anchors, cuda:0, exact):** TTT reconstruction bitwise per key + assay reproduction of
  the validator-exact endpoint values — sharpening s8 flank **0.8278569927266332**, center
  **1.189473623794586**, H **0.9907407760620117**; dampening s8 M **0.2960640796352685**, center
  **0.1436250155989566**, flank **0.4999147530512203**, H **0.1990740746259689**, rate
  **0.0493474886843507** (source: `v_ladder_s0p04.json`, coder evals; abs diff 0.0 expected on
  cuda:0). Non-8 seeds: ≤1e−6 vs the stored eval reports (G4-analog).
- **G3:** PPP identity under AMENDMENT-1 (both filenames, own-arm alpha metadata, state_dicts
  bitwise = pretrain; both M lookups bitwise equal).
- **G5:** per cell, `circ_raw` bitwise = declared source and recomputed k bit-consistent
  (seed-8 references: pretrain k to be recorded at G1 time from the pretrain checkpoint; α0.0
  +0.04727417230606079, α0.5 −3.5016465187072754 per training summaries).
- **Control-FB construction gates:** §2.3 tolerances; plus a null-edit gate in the original's
  spirit — rebuilding TTT via the control-splicing code path with FB:=trained-FB must be bitwise
  TTT (28/28-style, here 2 regimes × 4 seeds = 8/8).
- **Determinism repeat:** one cell (α0.5 TPT s8) assayed twice, exact match.
- **PPP-direct control:** the raw pretrain checkpoint run through all endpoints with no
  construction must equal the built PPP bitwise (EC1 analog), 4/4 seeds.

---

## 4. DEEPER STRATEGY ANALYSES (exact definitions + labeled predictions)

### 4.1 Per-set trained-vs-pretrain deltas
For each of the 7 tensors, per regime × seed: ‖Δ‖_F and ‖Δ‖_F/‖P‖_F (float64 on float32
values). SVD of Δ_hh (64×64) and Δ_fb (36×64): σ₁..σ₁₀ and top-5 energy e5 = Σ₁₅σ²/Σσ².
Comparison targets (original): α0.0 mass in W_ih/W_fb (‖Δ‖ 21–23), α0.5 in W_hh/circ_raw;
e5(Δ_hh): α0.5 ≈ 0.80, α0.0 ≈ 0.36 (WS P4). **Registered question:** does the surround shift
training mass — specifically, is the α0.0 ‖Δ_fb‖ (relative) SMALLER at s=0.04 than in the
original (the kernel doing flank work the FB rewrite previously had to buy)? Labeled
prediction: modest decrease or no change; a large decrease would itself be evidence that the
kernel absorbed part of the FB's old job.

### 4.2 FB alignment geometry
(i) Per-row cosine cos(w_i^T, w_i^P), i=1..36: median + fraction >0.9. (ii) Whole-matrix
normalized inner product ⟨W^T, W^P⟩_F/(‖W^T‖‖W^P‖). (iii) Cross-set coupling: E_proj =
‖Δ_fb V₅‖²_F/‖Δ_fb‖²_F with V₅ = top-5 right-singular vectors of that regime's Δ_hh (h-space);
analytic random-subspace null mean 5/64 ≈ 0.078. Tests whether the FB rewrite reads the subspace
the CELL rewrite writes — the co-adaptation geometry behind "trained FB out of context is
worse." (iv) Functional alignment: PCA of final-step h over the standard assay battery on the
pure arm; a_k = ‖W_fb u_k‖₂ for top-5 PCs, trained vs pretrain FB on the SAME (trained) h-PCs.
Labeled predictions: α0.0 — low row-cos (direction rewritten; WS A2-c), E_proj well above null;
α0.5 — high row-cos (FB barely moves; original Δ-norm), E_proj near null.

**Alignment-criticality (the dispatch's MORE-or-LESS question), causal form:** A_align(regime)
:= ρ_primary(TPT) − ρ_primary(TQT), per seed (primary = hit for α0.0, M for α0.5) — how much
worse a rotated trained FB is than a pretrain FB in the same trained context. Registered
statement: surround-sharpening FB is MORE alignment-critical than dampening FB iff
A_align(α0.0) > A_align(α0.5) in 4/4 seeds. Cross-study comparison to the original is
qualitative only (the original had no rotation control): the original FB premium on hit =
1 − ρ_hit(TPT) ≈ 0.51–0.58; the registered directional question is whether the s=0.04 premium
1 − ρ_hit(TPT) SHRINKS (kernel absorbed FB's flank role ⇒ FB matters less) or persists. Labeled
prediction: persists on hit, shrinks on the flank coordinate.

### 4.3 Gain/k configuration comparison
Table per regime × seed: g0..g4, k, som_margin (g1 − g2·g0), vs the no-surround originals.
Known anchors: s=0.04 s8 α0.0 k +0.0473 / margin 0.568 vs no-surround +0.0366; α0.5 k −3.5016 /
margin 1.833 vs −3.6932; pretrain +0.5457 (no-surround; s=0.04 pretrain k recorded at G1).
Registered question: same qualitative solution (small-positive vs deep-negative k) or a
surround-shifted gain configuration? Labeled prediction: same family, |k| slightly smaller in
both regimes (the fixed subtraction shoulders part of the suppression).

### 4.4 Surround-specific measurement-only questions (registered)
- **Q1 — does FB transplant alone now carry flank suppression?** In the original architecture
  FB-alone (PTP) was below baseline on every marker. At s=0.04 the surround drive is
  s·(f_pos @ Kᵀ): a fixed kernel converts ANY well-placed f into flank-band subtraction. ρ_flank
  (PTP) and ρ_flank(TTP) adjudicate; the s→0 counterfactual (§2.4) attributes. Labeled
  prediction: TTP flank ∂–F (trained CELL+FB produce trained-like f; only GAINS missing), PTP
  flank 0–∂ (trained FB on pretrain h may not produce a well-placed f), hit stays 0 for both —
  i.e. **flank suppression becomes more transplantable than placement**.
- **Q2 — is dampening still GAINS-locked with the surround present?** k=−3.50 dwarfs s=0.04, so
  removing GAINS (TTP) should still lose the phenotype and overshoot M above the host baseline
  as in the original. ρ_M(TTP) adjudicates; prediction: 0/overshoot repeats.
- **Q3 — softmax-temperature sensitivity of the dampening loop** (via R vs N): pfe =
  relu(36·softmax(W_fb h)−1) makes f scale-free but temperature-sensitive; R (≈half-norm FB)
  flattens the softmax and weakens the positive excess f that both k·f and the surround consume.
  Prediction: TNT carries (≈TPT); TRT partial-or-carries, ≤ TNT. If TRT < TNT cleanly 4/4, the
  "magnitude-preserving" qualifier in the baseline gloss is proven necessary, not decorative.
- **Q4 — CE-trip census repeat:** α0.5 PPT and PTT tripped 4/4 in the original; prediction: same
  class of trips at s=0.04 (the surround does not rescue FB/GAINS-only chimeras' competence).

### 4.5 Registered control hypotheses
- **H-C1 (dampening genericity):** α0.5 T·ctrl·T carries (ρ_M and ρ_center ≥ 0.75, no trip) for
  N and Q; R at least partial. Confirms the "k applies negative gain to any magnitude-sane
  direction" reading. Prediction: CONFIRM (medium-high confidence for N/Q, medium for R).
- **H-C2 (sharpening alignment):** α0.0 T·ctrl·T fails to carry hit (ρ_hit ≤ 0.25) for all
  three controls, and TQT ≤ TPT on hit (rotated trained FB no better than pretrain FB).
  Prediction: CONFIRM (the sharpened f must point at the right channels; a misplaced peaked f
  drives the kernel to suppress around the WRONG center — plausibly below TPT).

### 4.6 Predicted strategy map (labeled, falsifiable, per the established mechanism picture)
| Regime | Full carry | Partial | Absent/below | Novel prediction |
|---|---|---|---|---|
| Sharpening s=0.04 | TTT only | TPT (hit ∂ ≈0.4–0.5); TTP ∂–F on FLANK only | TPP, PTP, PPT, PTT (hit); PTT trip-fragile | flank dissociates from hit: kernel+f does flank; trio still owns placement |
| Dampening s=0.04 | TPT; TNT; TQT (controls) | TRT; PPT/PTT ∂-M but tripped | TPP, PTP; TTP overshoots | random-direction FB genuinely suffices at trained norm; GAINS remain the lock |
Each cell above is a prediction to be confronted, not a reading rule; misses are reported as
misses.

---

## 5. INVALIDATED / ADAPTED ORIGINAL CONVENTIONS (complete list, with reasons)
1. **Materials root:** donors live in scratch (ladder dirs), not the frozen `neuroips_runs` —
   G6 manifest gate substitutes for the original's frozen-md5 gate. READ-ONLY discipline binds.
2. **Frozen-M anchor value (G2-original):** 0.3320623037521497 belongs to the no-surround α0.5
   s8 arm — replaced by the s=0.04 validator-exact anchors (§3.6 G0). The old anchor may still
   be run once as an evaluator-sanity check (the frozen evaluator itself is unchanged).
3. **ρ_rate as the dampening adjudicator:** demoted to raw-only (original S1/EC2: UNREADABLE
   4/4; same collapse expected at s=0.04 — donor rate_A 0.044–0.052 sits in the original's
   collision zone with the pretrain value). Replaced by ρ_M + ρ_center (§3.2).
4. **Profile coordinates (center/flank ratios) added** to the transplant grid — they are the
   surround phenotype's registered metrics (flank study conventions) and did not exist in the
   original grid. New floors pre-registered (§3.3).
5. **FB controls:** not inherited (none exist) — new constructions (§2.3), flagged to the lead.
6. **Host config:** host must carry s=0.04/σ=4.0 (mechanism present at measurement); the
   surround is therefore ACTIVE in the host, which changes denominator expectations (§3.3 risk).
7. **Donor family status:** s=0.04 is O2 (not a confirmed family) — caveat recorded (§2.1);
   within-seed ρ readings remain valid.
8. Unchanged inheritances for completeness: 2³ matrix, splicing mechanics, AMENDMENT-1 PPP,
   CE gate, ρ bands/floors (plus new ones), house rules, gate-chain style, cuda:0 bitwise
   envelope, sequential one-checkpoint-at-a-time del+gc, PYTHONHASHSEED=0 python3 -B.

## 6. RISKS / UNKNOWNS (registered before measurement)
- **R1:** sharpening flank denominator may floor out (§3.3) — fallback registered (raw flank +
  s→0 counterfactual + ρ_center/ρ_hit adjudication).
- **R2:** control-FB cells may trip CE in the α0.0 factorial (the original PTT precedent) —
  house rule handles; predictions in §4.5 are conditioned on untripped seeds.
- **R3:** hybrid M via the frozen evaluator requires dual-filename dirs per cell (inherited
  mechanics); the evaluator validates alpha metadata — Phase 2 must write per-arm metadata
  exactly as AMENDMENT-1 (probe precedent: shared-bytes construction FAILS with ValueError).
- **R4:** seeds 9/10 dampening donors are sub-band on M (O2) — strategy claims generalize to the
  solution family, not the in-band phenotype (§2.1); if seed-level dampening carries disagree
  along the in-band/sub-band split (8,11 vs 9,10), that split is reported as a finding, not
  averaged away.
- **U1 (open):** whether pretrain-host flank_ratio at s=0.04 lands 0.85–0.97 (predicted) is
  unmeasured until Phase 2's PPP row — first number to check before reading any hybrid flank ρ.

## 7. PHASE-2 EXECUTION SKETCH + VALIDATION POINTERS
Order: G6 manifest → G1/G2 (partition + pretrain equality, 8 arms) → build 60 core nets (G0,
G3, G5, construction audit) → core assays E1/E2/E3 + profile (64 cells) → control builds
(construction gates + null-edit 8/8) → control assays (30 cells) → s→0 re-assays (§2.4) → Δ/
geometry/gain analyses (§4.1–4.3, pure post-processing of checkpoints + stored assay states) →
synthesis tables (ρ + classes + predictions confronted). Writes only under
`/home/vishnu/scratch/transplant_surround_20260823/` and this study dir; sequential; cuda:0.
Validator (Phase 3): independent re-derivation of the §2.5 load-bearing cells from raw
checkpoints (own splicing + own assay), the two CE-trip censuses, one control-FB rebuild from
the pinned seeds, and the G0 anchor chain; GO/NO-GO on the strategy map as read by the §3 rules.
