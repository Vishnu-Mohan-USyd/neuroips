# VALIDATION REPORT — flank-suppressed sharpening, Phase 3 independent audit
Validator, 2026-08-19 (AEST). Study dir: `/home/vishnu/neuroips_analysis/flank_sharpening_20260819/`.
Claim under test: enabling the frozen lib's feedback-recruited subtractive surround via two config
constants only (`pred_inhib_strength` 0.0→0.05, `pred_inhib_sigma_channels` 0.65→4.0) yields, on
the standard frozen assay: flank_ratio ≤0.85, center_ratio ≥1.15, H ≥0.95, with the s→0 inference
counterfactual removing the flank suppression (A4).

All numeric evidence below was produced by MY OWN code end-to-end
(`/home/vishnu/neuroips_outputs/validator_flank_20260819T013110Z/`): own stimulus construction
from the registered convention (asserted `torch.equal` against the frozen registered generator),
own alignment/binning/ratio/H/vitality implementations, own A4 counterfactual orchestration, own
canonical `state_sha256` implementation. The coder's eval scripts were read for convention
comparison only, never executed as an evidence path.

---
## Verdict: GO

All four pre-registered acceptance criteria pass on the official seed 8, and on every
confirmation seed (9/10/11), with zero reinterpretation. Stretch goal (flank ≤0.75) not met on
any seed, as reported. Four minor, non-blocking issues listed at the end.

---
## Check 1 — independent endpoint re-derivation: EXACT MATCH, all four seeds

My pipeline vs the coder's reports, compared at full float precision (`==`, not tolerance):

| seed | H (bar ≥0.95) | center (bar ≥1.15) | flank (bar ≤0.85) | vitality | A4 flank (s→0) | A4 H | A4 center |
|---|---|---|---|---|---|---|---|
| 8 (official) | 1.0 | 1.1923348866420662 | 0.7886317366506529 | pass | 0.9715966626670492 | 0.944444477558136 | 1.2334376880427118 |
| 9 | 0.9814814925193787 | 1.203521534899704 | 0.7873379749128777 | pass | 0.969569162933151 | 0.8981481790542603 | 1.245438999989701 |
| 10 | 0.9953703880310059 | 1.2159847436745765 | 0.7863693137382797 | pass | 0.9680484629401713 | 0.875 | 1.2572708374035435 |
| 11 | 0.9953703880310059 | 1.215624266177482 | 0.7857501779234405 | pass | 0.968385348119569 | 0.8888888955116272 | 1.2560317300793773 |

TOTAL MISMATCHES vs coder reports across all seeds, all quantities (incl. all vitality-band
values): **0**. Flank ratio spread across seeds 0.786–0.789 (0.003) — strongly seed-invariant.
A4: s→0 at inference returns flank to 0.968–0.972 (≈ no-surround reference; frozen healthy cell
flank context ~0.97) on every seed while center stays 1.23–1.26 — the surround, and nothing else,
does the flank work. Vitality (A3, |offset|≤10° > 0.01): seed-8 band 0.558–1.392, pass on all
seeds. Frozen-cell H reference 0.9954 (t_battery): the mechanism costs nothing at endpoint
(seed 8 H = 1.0).

## Check 2 — config/diff chain: CLEAN
- Study harness sha256 = `9db8f975531b55a86c54791c68908708403cd4df72a97591ce8199b1ec25937e`,
  equal to the RUN_LOG-audited post-edit sha. My own diff vs the frozen harness
  (`heatmap_sweep_20260818/harness/train_sweep.py`): exactly the two constant lines 45–46.
  Import symlinks point into the frozen READ-ONLY repo.
- Mechanism config-carried in TRAINING: `run_start` model_config records 0.05/4.0;
  `common_pretrain_final.pt` tuned_net_config carries 0.05/4.0 (mechanism present from pretrain
  step 0); `alpha_0p0_final.pt` carries 0.05/4.0.
- Mechanism config-carried in MEASUREMENT: the surround kernel is a `persistent=False` buffer,
  absent from state_dict (verified), rebuilt from checkpoint config at load — measurement uses
  exactly the trained mechanism.
- `freeze_local_comp`: local_comp_strength_raw bytes-equal pretrain→final (my own check).
- run_start vs frozen sweep cell `s8_t1p0_e0p0`: model_config identical except the two constants;
  task_weight null resolves to 1.0 (= frozen), and the final checkpoint payload records the
  resolved 1.0. Seeds 9/10/11 run_starts differ from seed 8 only in `seed` and `wall_time`.

## Check 3 — A/A CONTROL: BITWISE PASS (pre-authorized retrain)
- σ-inertness at s=0 proven FIRST, in the frozen lib (`v_sigma_inert.py`): two nets, identical
  weights, σ=0.65 vs 4.0, s=0 both — kernels differ (max|Δ| 0.514) yet full-assay forward outputs
  bytes-identical, all parameter gradients bytes-identical, and gradients into the `fb`/`l4`
  inputs (the W_fb training path) bytes-identical. σ therefore kept at 4.0 per dispatch condition.
- Control harness (`/home/vishnu/scratch/validator_flank_aa_20260819/harness/`): byte-copy of the
  study harness, strength 0.05→0.0. Diff vs frozen original = σ line ONLY; diff vs study harness
  = strength line ONLY — a single-variable off-switch of the mechanism.
- Launch conditions: GPU idle (0% util after coder's seed-11 completed), MemAvailable 86.9 GB
  ≥ 25 GB, harness verified to set `use_deterministic_algorithms(True)` + `cudnn.benchmark=False`
  + full seeding; exact study invocation, seed 8; 8 GB RSS watchdog (never triggered; RSS ~1.9 GB);
  exit 0.
- RESULT (`v_aa_compare.py`, own state_sha256):
  - final state sha `afc2d7ae5351431fab4b481714305b1b6c570d627764f9102733869072c702f4`
    == frozen cell recomputed-from-disk == t_battery `final_state_sha256`. **BITWISE.**
  - pretrain state sha `926c53fb00025679b24aa837b2252a6ddc21cba32f777619c6df9bfab4ceb574`
    == frozen parent recomputed == t_battery `parent_state_sha256`. **BITWISE.**
  - Recursive payload tree-diff of the ENTIRE final checkpoint (state, optimizer state, RNG
    generator states, references, metadata): exactly ONE differing leaf —
    `tuned_net_config.pred_inhib_sigma_channels` 4.0 vs 0.65. File-level shas differ solely for
    that reason (expected by design; documented).
- Implications: (a) the study harness contains NO hidden behavioral deltas beyond the two
  constants — an 11,000-step training trajectory is bitwise-reproducing the frozen cell when the
  strength is zeroed; (b) σ is end-to-end inert at s=0; (c) `pred_inhib_strength` alone carries
  the entire effect.
- Benign artifact: the frozen cell's historical shared-pretrain payload lacks a `task_weight`
  key that current pretrain payloads include (writer-version difference in that one archived
  file); pretrain STATE bitwise identical, and final-payload keysets identical.

## Check 4 — criterion-4 mechanism audit: PASS
- **Objectives/assay untouched:** proven by the diff chain — any objective or assay edit would
  appear in the 2-line diff; none does. Objective remains task_weight·task + α·energy with α=0;
  no term references profile shape or bands.
- **Trained INTO the network:** config present from pretrain step 0 through arm (Check 2);
  measurement runs the same config. Corroborating: removing the surround at inference DEGRADES
  the task on trained weights (H 1.0→0.944 seed 8; 0.875–0.898 on seeds 9–11) — the weights
  co-adapted to the surround; it is a load-bearing part of the trained circuit, not a bolt-on.
- **A4 counterfactual:** reproduced by MY pipeline on all four seeds (table above).
- **Biological grounding (primary full texts in `papers/`, both load-bearing claims):**
  - *Adesnik 2012* — every cited statistic verified verbatim in the full text: PC preferred size
    22±2°; SOM SI 0.09±0.06 (0/8 cells significant — "completely lacked surround suppression");
    SOM preferred size 86±3° (~4× broader); L4→SOM excitation 17±5% of PC vs horizontal L2/3
    241±85%; SOM photo-hyperpolarization reduces surround suppression 30±10% (p<0.00022),
    facilitates larger-than-preferred responses 74±19%, leaves preferred-size responses unchanged
    (−7±7%, p>0.45) — the center-sparing, surround-targeting incidence claim is faithful.
  - *Zhang 2014* — verified verbatim: modulation factor +0.17±0.02 at 0 μm (p=4e−16, n=152) and
    −0.15±0.03 at 200 μm (p=4e−6, n=78); SOM+ inactivation "converting the surround suppression
    into a slight facilitation"; VIP+ "selectively enhance the responses at 0 μm through localized
    inhibition of SOM+ neurons"; note 37 verbatim ("a magnification factor of 10 μm/° in mouse V1
    … 200 μm of cortical distance corresponds to 20° of visual angle") — the σ = 4-channel anchor
    is faithful; the discussion's "operating … in stimulus feature space" transplant license is
    verbatim.
  - Honest scope note: the cited biology is a SPATIAL surround; the model instantiates it in
    orientation space. DESIGN.md states this transplant explicitly and grounds it in Zhang's
    feature-space proposal and the Ben-Yishai ring-model lineage — clearly labeled, not hidden.

## Check 5 — gate rulings review (G-R1, G-R2)
- **G-R1 (flank "trending down" waived as misfit): AGREE.** My own reading of
  `gate_trend_full.json`: flank 0.7711 at arm step 250, 0.7848 at 4000 — full-magnitude
  suppression from the first snapshot, never descending, because the mechanism is fixed
  connectivity carried by config from pretrain (I verified the pretrain checkpoint carries it).
  A criterion demanding a developing trend cannot bind a mechanism whose effect is structurally
  present from step 0; the anti-absent/flat purpose is served instead by the A4 counterfactual
  (mechanism demonstrably does the work) and by flank <1 and center >1 at every one of the 16
  snapshots (both verified). What must develop — circuit health around the fixed mechanism — did:
  H 0.486→0.884 across the gate window, 1.0 at endpoint.
- **G-R2 (H 0.884 vs 0.9 gate bar, pass on intent): AGREE with the conclusion; one factual
  correction to its stated basis.** The H trajectory is NOT strictly monotone: 16 snapshots
  0.4861, 0.5324, 0.6065, 0.5926, 0.6713, 0.7130, 0.7685, 0.7963, 0.8009, 0.8194, 0.8333,
  0.8194, 0.8519, 0.8519, 0.8796, 0.8843 — two local dips (750→1000 and 2750→3000, each −0.0139
  = 3 of 216 histories; H quantum 1/216 ≈ 0.0046) and one flat pair. The substantive basis is
  intact: net rise +0.398, the gate value 0.8843 is the trajectory maximum, endpoint 1.0, and a
  0.016 shortfall on a rising trajectory does not meet the bar's kill-collapsing-runs purpose.
  "Monotone rising through all 16 snapshots" (ruling text and RUN_LOG) should read "rising trend,
  maximum at the gate, two 3-history local dips".
- Both rulings confined to the mid-run gate; the ACCEPTANCE criteria passed literally on all
  seeds — confirmed by Check 1.

## Issues (all minor; none blocks GO)
1. "Monotonically rising across all 16 snapshots" in G-R2/RUN_LOG overstates (two 3-history
   dips) — severity: minor — next action: correct the wording in archival docs; conclusion
   unaffected.
2. `effective_k` context value differs between my CPU softplus (0.04994739501287171) and the
   coder's GPU softplus (0.04994745301448633) — a 1-float32-ulp class difference in a
   context-only quantity; all criterion numbers match bitwise — severity: minor/cosmetic.
3. DESIGN §6 predicted flank suppression "deepening as f sharpens"; observed flat/slightly-up
   (0.771→0.789). The "immediate onset" half of the prediction was right; the trend direction was
   not. Not load-bearing (criteria are endpoint bars) — severity: minor — note for the paper's
   mechanism narrative.
4. Frozen cell's archived shared-pretrain payload lacks the `task_weight` key present in current
   payloads (writer-version artifact; states bitwise identical) — severity: minor/informational.

## Multi-seed status
Fully covered here: seeds 9/10/11 endpoints and A4 counterfactuals re-derived with my own
pipeline, zero mismatches; per-seed configs carry 0.05/4.0; run_starts differ from seed 8 only in
seed/wall_time. Nothing further needed from the coder for the multi-seed verdict.

## Conduct proof
`readonly_proof_flank.txt` (in my output dir): frozen roots (repo, neuroips_runs, heatmap sweep
cells + analysis, orientation figs) — ZERO files newer than my session marker (11:31:10). My
writes confined to `/home/vishnu/neuroips_outputs/validator_flank_20260819T013110Z/`, the
pre-authorized control dir `/home/vishnu/scratch/validator_flank_aa_20260819/`, and this
VERDICT.md (explicitly authorized). Newer files in the study analysis dir = the lead's PROTOCOL.md
update (11:44:29, multi-seed log entry); newer files in the study scratch dir = the coder's
concurrent seeds-9/11 workflow — none mine.

## Artifacts (validator output dir)
- `v_flank_eval.py` + `v_eval_predinhib_s0p05_sig4_seed_{8,9,10,11}.json` — independent endpoint
  evals, all seeds
- `v_sigma_inert.py` / `v_sigma_inert.json` — σ-inertness proof at s=0 (forward + gradients)
- `v_aa_compare.py` / `v_aa_compare.json` — bitwise A/A gate result
- `v_curve_check.json` — 25-point curve verification (addendum below)
- `readonly_proof_flank.txt` — conduct proof
- Control run: `/home/vishnu/scratch/validator_flank_aa_20260819/runs/aa_s0/seed_8/` (exit 0)

## Addendum (2026-08-19, on lead's multi-seed forward): 25-point figure curves verified
The lead's multi-seed message (crossed with this report's delivery) requested spot-verification of
at least one non-8 seed end-to-end; seeds 9/10/11 were already fully re-derived above (Check 1,
zero mismatches). Supplementary closure for the figure phase: the full 25-point aligned profile
curves (`curve_adapted` and `curve_baseline_t0`, offsets −60°…+60°) were recomputed with my own
alignment pipeline for the official seed 8 and spot-pick seed 9 and compared elementwise to the
coder's endpoint reports at full float precision: **25/25 exact on both curves, both seeds;
offset grids identical**. The figure build's inputs are verified at every point, not only at the
criterion bins. GO verdict unchanged.

---
# Addendum 2 (2026-08-19): Phase-4 / s=0.04 joint-ladder multi-seed verification

**Verdict on the FAMILY claim (one fixed architecture s=0.04/σ=4.0, both regimes, all seeds,
pre-registered bands verbatim): NOT CONFIRMED — Outcome O2.** Seed-level joint verdicts confirmed
exactly as reported: **8 YES / 9 NO / 10 NO / 11 YES.** The sole failing bar anywhere is
dampening P3_M on seeds 9 and 10; every other bar passes on every seed in both regimes.

Method: my own measurement core (own stimuli + alignment, the same pipeline proven equal to the
frozen generator in Check 1), extended with the frozen M definition
(whole-36-bin expected-A AUC / t0 AUC), **E0-anchored before any new cell was read**: on the
frozen original α0.5 seed-8 checkpoint (file sha verified = 156cc0f2…), my values vs the FROZEN
gate-decision artifact: Fret / M / rate_A / mean_rate_t0 abs diff **0.0**, Cret 2.8e−17 — PASS.
My reference values equal the coder-pinned reference on every key, and my re-derived band edges
(0.85/1.15 × my anchored reference) are **bitwise equal** to the pinned bands:
M ∈ [0.2822529581893272, 0.3818716493149721], H ∈ [0.16527778059244155, 0.22361111491918562].
All eight s=0.04 cells re-derived (six new + the two debugger-verified seed-8 cells first-hand);
**zero exact-precision mismatches vs the coder's eval reports on any compared quantity.**
Artifacts: `v_ladder_eval.py` / `v_ladder_s0p04.json` in my output dir.

SHARPENING (α0.0) — bars flank ≤0.85, center ≥1.15, H ≥0.95, vitality; ALL FOUR SEEDS PASS:
| seed | flank | center | H | vitality |
|---|---|---|---|---|
| 8 | 0.8278569927266332 | 1.189473623794586 | 0.9907407760620117 | pass |
| 9 | 0.8252633915461819 | 1.213162208090845 | 0.9675925970077515 | pass |
| 10 | 0.8245212776501858 | 1.220085032291537 | 0.9907407760620117 | pass |
| 11 | 0.8240432954679329 | 1.221555421152803 | 0.9861111044883728 | pass |

DAMPENING (α0.5) — P1 ≤0.35, P2 center<flank, P3 H/M in band, P4 rate>0.01 + A3 band alive:
| seed | M (band floor 0.2822529581893272) | P3_M | center | flank | H | rate | P1/P2/P3_H/P4 |
|---|---|---|---|---|---|---|---|
| 8 | 0.2960640796352685 | PASS | 0.1436250155989566 | 0.4999147530512203 | 0.1990740746259689 | 0.0493474886843507 | all pass |
| 9 | 0.2637179371185669 | **FAIL** (short 0.01853502107076027, −6.6% rel) | 0.0927245440588919 | 0.4929806120314839 | 0.2037037014961243 | 0.0439560852294238 | all pass |
| 10 | 0.2820059371106222 | **FAIL** (short **0.0002470210787050009** = 0.0875% of the floor) | 0.1289570736084887 | 0.5032892060579721 | 0.2129629701375961 | 0.0470042999057167 | all pass |
| 11 | 0.3090770353323293 | PASS | 0.1639250766172052 | 0.5109724199079152 | 0.1759259253740311 | 0.0515164673892370 | all pass |

The pivotal number is CONFIRMED at full float precision: seed-10 dampening
M = 0.2820059371106222 vs floor 0.2822529581893272 → below the band by 2.470210787050009e−4.
This is a real sub-band value under the pre-registered bar, not a rounding artifact; the bar is
inclusive and the value is strictly below it. Seed 9's 0.2637179371185669 is unambiguous.

Config/provenance — CLEAN on all 8 cells (own state_sha256 implementation):
- s=0.04/σ=4.0 carried in run_start, pretrain checkpoint config, AND final checkpoint config of
  every cell; step 8000; alpha/task_weight correctly 0.0/1.0 and 0.5/0.5; seeds correct.
- Identical-pretrain signature correct: within each seed the α0.0 and α0.5 fresh pretrains are
  state-bitwise IDENTICAL (α touches only the arm), the four seeds' pretrains are distinct, each
  differs from the same seed's s=0.05 pretrain (mechanism-in-pretrain dose signature), and all 8
  final states are distinct. freeze_local_comp bytes-equal pretrain→final in every cell.
- Coder eval-report checkpoint paths match the cells I measured.

Family-verdict statement (plain, per O2): under the pre-registered bands, with no band loosened,
the s=0.04 configuration does NOT confirm the family claim — 2 of 4 seeds fail the dampening M
band (one by 6.6% relative, one by 0.09%). The sharpening regime is uniformly robust at s=0.04
(flank spread 0.0038 across seeds). Context recorded by the debugger (labeled, not re-verified by
me): the original no-surround α0.5 family's own seeds span M 0.3071–0.3321 with the band's seed-8
anchor at the top of that range, and clear the floor by margins (0.025–0.031) comparable to the
demonstrated between-run scatter — calibration context for the user's decision, not a
modification of the verdict. Any distribution-referenced re-test requires fresh seeds under a
newly pre-registered criterion, per the outcome rules.
