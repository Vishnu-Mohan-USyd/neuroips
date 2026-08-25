# VALIDATION REPORT — transplant_surround_20260823 (Phase 3, validator)
2026-08-23. Dispatch: lead. Governing rules: DESIGN.md (applied verbatim; sections cited per
check) + PROTOCOL.md anchor ruling (artifact floats loaded programmatically, never document
decimals). Evidence artifacts (mine): `/home/vishnu/neuroips_outputs/validator_transplant_20260823T065540Z/`
{v_tp_eval.py, v_tp_raw.json, v_tp_rho.py, v_tp_verdict.json, v_tp_eval.log}.

---
## Verdict: **GO** on the strategy map as stated in PROTOCOL's Phase-2 entry

Every load-bearing claim reproduced from raw donor checkpoints with my own splicing and my own
measurement pipeline end-to-end. Three notes for the user report (none blocks): N-1 a range typo
in the PROTOCOL headline, N-2 a seed-9 caveat on the premium wording, N-3 my ruling on the
disclosed N5 mid-run correction (fallback-consistent; the load-bearing number verified
first-hand regardless).

## Independence and anchoring
- My pipeline: own stimulus battery, own alignment, own profile/hit/CE metrics (the core the
  flank study proved at abs diff 0.0), own hybrid splicing from the raw donor checkpoints, own
  FB-control construction from the pinned RNG seeds, own float64 k/SVD/E_proj. The coder's
  scripts were read only to pin conventions; none were executed as my evidence path.
- E0 anchor PASS 8/8 before any new cell was read: my re-assay of every donor endpoint is
  bitwise equal to my own sha-pinned prior artifact `v_ladder_s0p04.json` (sha
  8b1a9299d68c1a35d6a80d4d34f848dc0fda505745f97dafe665de9c5af723bc), all metrics including
  vitality bands.
- Donor integrity (mine): all 8 arm states + 4 pretrains match my prior recorded state shas;
  seed-8 pretrain state sha = 4c5b1a32… (the recorded anchor); within-seed pretrains identical
  across regime dirs 4/4; every config carries s=0.04, σ=4.0 (arm + pretrain + every hybrid).
- My G1-analog partition proof 8/8 arms: pretrain→arm diff confined to the 7 CELL/FB/GAINS
  tensors; the 5 non-component keys bitwise identical in every arm.

## Evidence (commands → results)
- v_tp_eval.py (cuda:0, PYTHONHASHSEED=0 python3 -B, MemAvailable-gated, sequential) →
  **50/50 table cells: my hybrid state sha == coder hybrid file state sha AND every compared
  endpoint bitwise equal to n4_assay.json** (H, center_ratio, flank_ratio, M_auc_ratio,
  continuation_mean_rate, mean_rate_t0, vitality band+pass, profile min/max, hit
  (=registered_final_A_on_y), CE_A, CE_B, trip flag, k). Cells = 46 distinct nets: seed 8
  sharpening {PPP,TTT,TPT,TTP,TQT,PTP,TRT,TNT,PPT} + dampening {TTT,TPT,TNT,TRT,TQT,TTP,PPT,
  PTT}; seeds 9/10/11 {PPP,TTT,TPT,TQT}×α0.0 (+PPT s9) and {TTT,TPT,TNT,TQT,TTP}×α0.5
  (+PTT s9). Exact-match bar met with zero mismatches.
- Frozen-evaluator M equivalence: |E1 M − my continuation/t0 M| ≤ 2.22e−16 on all 50 cells.
- v_tp_rho.py → my ρ vs n7_synth.json: max |Δρ| = 8.9e−16 over every compared coordinate; band
  agreement 100% (string formatting differs only: my "0,below_baseline"/"d" vs coder
  "0_below_baseline"/"partial").
- FB controls rebuilt by me from the pinned seeds (manual_seed 20260823 → R via
  build_tuned_from_config(host cfg) on CPU; manual_seed 20260824 → float64 Gaussian → QR →
  sign-fix) → **R, Q64, and all 8 arms' N and Q tensors bitwise equal to controls_fb.pt**; my
  gates: ‖QᵀQ−I‖∞ = 6.4e−16 < 1e−5, N rel-norm errors < 1e−6, Q Frobenius rel err < 1e−5 — all
  PASS. (DESIGN §2.3 is silent on the Gaussian/QR dtype; the float64 choice is recorded, gates
  pass — no divergence between registration and construction that matters.)
- s→0 re-assays (my own config-override rebuilds, α0.0 seed 8 {PPP,TTT,TPT,TTP,PTP}) → s0 flank
  bitwise equal to n5_s0.json; Δflank(s−s0): TTP **−0.1500** vs PTP **−0.0633** (headline
  confirmed), TTT −0.1452, TPT −0.1317, PPP −0.0917.
- k tables (my float64 softplus on circ_raw, incl. the no-surround originals recomputed from the
  frozen S2 checkpoints) → all bitwise equal to n6; pretrain k = +0.545719 in BOTH architectures.
- E_proj (my float64 SVD from raw checkpoints) → abs diff 0.0 vs n6 on all 8 arms.
- Read-only conduct: `find <scratch study root, donor dirs, neuroips_runs, study dir> -newer
  SESSION_START.marker` → empty (before this VERDICT.md write; writes confined to my outdir).

## The strategy map, claim by claim (my numbers)
**DAMPENING (α0.5).**
- TPT carries 4/4: ρ_M = 0.9786/0.9267/0.9630/**1.0044**, ρ_center = 0.9958/0.9581/0.9957/
  1.0268 (s8/9/10/11), untripped, denominators readable (M den −1.05..−1.09, center −1.63..
  −1.72). CONFIRMED. → **N-1:** PROTOCOL's headline range "ρ_M 0.927–0.986" is a transcription
  slip; the artifact (and my) range is **0.927–1.004** (TABLES' own H-C1 line has it right).
- **H-C1 REFUTED — verified.** No control carries on any seed. My own numbers: TNT ρ_M 0.4785/
  0.4737/0.4898/0.5377, TQT ρ_M 0.4589/0.5851/0.4805/0.4896 (ρ_center 0.54–0.62), TRT s8 ρ_M
  0.5610 (s9–11 from n7, 0.558–0.611, cross-checked); zero trips. The registered prediction (N
  and Q carry) fails on all four seeds on both primaries — the refutation rides entirely on
  cells I re-derived first-hand. Magnitude-matched or rotated FB ≠ pretrain FB: dampening needs
  a task-meaningful FB direction.
- Q2 GAINS-removal overshoot repeats 4/4 (mine): M(TTP) 1.6210/1.6593/1.6655/1.5853 > M(PPP)
  1.3554/1.3541/1.3570/1.3565; ρ_M −0.25/−0.28/−0.29/−0.22 (below baseline).
- Q4 trip census (my CE, 50 cells, 100% agreement with coder census): PPT α0.5 s8 max CE_A
  11.5898 > 10.7506 TRIP; PTT α0.5 s8 = 10.5862 **untripped** (just under the gate) while s9 =
  13.7624 TRIP — the "PTT trips 3/4, not s8" finding confirmed.
**SHARPENING (α0.0).**
- Flank UNREADABLE 4/4 — R1 condition audit (lead's question): my flank denominators
  +0.017642/+0.016031/+0.010670/+0.001992, ALL < 0.05 floor. U1 MISS confirmed: host flank
  0.8102/0.8092/0.8139/0.8221 sits at TTT level (0.8279/0.8253/0.8245/0.8240), below the
  predicted 0.85–0.97 band. The registered R1 fallback was therefore correctly invoked on a
  condition that genuinely fired on all four seeds.
- TPT hit partial 4/4 (mine): ρ_hit 0.3604/0.5357/0.4091/0.3684 (raw hit 0.6620/0.7269/0.6898/
  0.6528), within the predicted 0.4–0.5 band loosely (s9 above at 0.536, still partial band).
  ρ_center(TPT) 1.08–1.11 (center fully transplanted).
- FB placement premium: 1−ρ_hit(TPT) = 0.6396/0.4643/0.5909/0.6316 vs original range
  [0.51, 0.58]. → **N-2:** "PERSISTS" is the right answer to the registered directional
  question (did the premium shrink? — no: 3/4 seeds sit above the original's whole range;
  comparison registered as qualitative-only), but s9's 0.464 sits below that range and should
  ride along in the user report (TABLES already reports it honestly).
- **H-C2 CONFIRMED — verified.** Every resolvable control seed has ρ_hit ≤ 0.25 — in fact all
  are far below baseline: my own cells TQT −0.8468/−0.8393/−0.8818 (s8/9/10; s11 TRIP, max CE
  12.8702, mine), TRT s8 −0.8559, TNT s8 TRIP (max CE 11.8092, mine); coder-corroborated
  rest: TNT s10/s11 −0.8636/−0.7544, TRT s9/10/11 −0.759/−0.873/−0.746, TNT s9 TRIP. TQT ≤ TPT
  on hit 3/3 resolvable (−0.85 vs +0.36, −0.84 vs +0.54, −0.88 vs +0.41), s11 UNRESOLVABLE by
  TQT trip — house rule applied as registered. Misaligned trained FB drives placement BELOW the
  untrained baseline (my ρ range −0.75..−0.88).
- NOVEL center/placement dissociation: PPT (GAINS alone) ρ_center 1.2736/1.2874 (mine s8/s9;
  1.3124/1.3117 coder s10/s11) with ρ_hit −0.74/−0.59/−0.65/−0.58 — center carries, placement
  goes below baseline, 4/4.
- Q1 kernel attribution (mine): PTP raw flank 1.0393 > host 0.8102 (FB-alone does NOT suppress
  — raises flank); Δflank(s−s0) TTP −0.1500 vs PTP −0.0633 — flank suppression is kernel work
  given a well-placed f. Confirmed bitwise against n5 and re-derived from my own rebuilds.
**CROSS-REGIME.**
- A_align (mine, bitwise = coder): α0.0 1.2072/1.3750/1.2909/UNRESOLVABLE vs α0.5 0.5197/
  0.3416/0.4825/0.5148 — sharpening more alignment-critical 3/3 resolvable seeds. Confirmed.
- E_proj INVERTS the registered prediction — verified (my float64 SVD, diff 0.0): α0.0
  0.0541/0.0469/0.0487/0.0498 all BELOW the 5/64 = 0.0781 null; α0.5 0.3155/0.2645/0.2608/
  0.2519 all ABOVE it.
- k: s=0.04 α0.0 +0.0473/+0.0525/+0.0540/+0.0542 LARGER in |k| than originals +0.0366/+0.0454/
  +0.0482/+0.0454 (4/4); α0.5 −3.5016/−3.3001/−3.2608/−3.3540 SMALLER in |k| than originals
  −3.6932/−3.9058/−3.4653/−3.5278 (4/4); pretrain +0.5457 both architectures. All my values
  bitwise equal to n6, originals recomputed by me from the frozen no-surround checkpoints.
- No systematic 8/11-vs-9/10 split: consistent with my numbers (dampening TPT carries on both
  in-band and sub-band seeds; only band-edge letter wobbles in n7's split table).

## N-3 — ruling on the disclosed N5 mid-run correction (lead's question)
The corrected selection is **the registered fallback, not a post-hoc choice**, on three grounds:
1. The floor condition fired for real: my own α0.0 flank denominator +0.017642 < 0.05 (all four
   seeds) — §3.3 says below floor ⇒ UNREADABLE, *no adjudication on that coordinate*. The first
   run's selection read ρ_flank off that floored denominator — that was the rule-INCONSISTENT
   selection, rightly superseded (both runs' provenance kept, disclosed in RUN_LOG + n5_s0.json).
2. The corrected α0.5 rule-extras are exactly what §2.4's registered rule yields on MY ρ_flank
   values: qualifying cells {TPT 0.897, PPT 0.476, PTT 0.470, TRT 0.341} minus the
   already-registered set {TTT,TPT,PPT,PTP,PPP} = **{PTT, TRT}** = the coder's selection (2 ≤ 8
   bound).
3. The α0.0 "full descriptive set" (9 cells) is licensed by the pre-registered R1 fallback
   ("raw flank + the s→0 counterfactual" as the descriptive answer) — the §2.4 selector cannot
   operate on an unreadable coordinate. Caveat stated plainly: a literal reading of §2.4's
   enumeration would not have produced TTP-α0.0's s→0 measurement, and the Q1 headline uses it;
   R1 (registered pre-measurement, with the floor risk labeled and its condition met) is the
   license. Materially: I re-derived that number bitwise from my own splice+override rebuild
   (TTP Δflank −0.1500), so the claim does not rest on the coder's selection at all.

## Scope notes
- decode/rate markers were NOT re-derived (decoder-training pipeline): under §3.2 no load-bearing
  claim rides on them (rate demoted to raw-only; decode is a companion). Every primary
  (hit/M/center/flank/CE) was re-derived first-hand.
- Two k renderings exist (float64-on-raw vs float32 in-graph, e.g. −3.50164602 vs −3.50164652);
  tables use float64-on-raw consistently — no issue.
- H-C1/H-C2 cells outside my 46-net matrix (TRT α0.5 s9–11; TNT/TRT α0.0 non-s8 seeds) are
  cross-checked against n7 rather than first-hand; the verdicts do not change under any of them
  (H-C1's refutation is decided by my own N/Q cells; H-C2's is decided by my own TQT/TRT/TNT-s8
  cells plus coder cells that only add further-below-baseline values), and the coder pipeline's
  fidelity is established by 50/50 bitwise agreement on my matrix.

*Validator, 2026-08-23. GO.*
