# Research Log

## Conventions

- Reverse chronological (newest first).
- One entry per significant milestone — not per commit.
- Each entry: **Context** → **What was tried** → **Outcome** (with concrete numbers) → **Decision/next step** → **Pointers** (commits, docs, figures).
- Entries are kept short (≤ 20 lines) and link out to the full writeups for detail.

## Active branches

| Branch | Purpose | Latest commit |
|---|---|---|
| `main` | Stable baseline | `4c57d47` (2026-03-31) |
| `single-network-dual-regime` | Dual-regime trained baselines (Network_mm, Network_both, Fixes 1–4) | `8e13d69` (2026-04-12) |
| `failed-dual-regime-experiments` | Architectural rescue chain R1+R2 / R3 / R4 / R5 | `39375ac` (2026-04-13) |
| `dampening-analysis` | Re-centered tuning analysis + figures + corrective docs; R1+R2 set as canonical default; Decoder C ex/unex eval; cross-decoder matrix | `37001a1` (2026-04-17) |

---

## 2026-04-22 — Cross-decoder comprehensive matrix + legacy reference networks (Tasks #23–#26)

**Context:** After Tasks #19–#22 showed R1+R2 is *decoder-robust sharpening* on the paired-fork paradigm and *decoder-robust dampening* on observational paradigms (see 2026-04-19 entry below), we still lacked a single audit table cross-cutting every ex-vs-unex claim in the project. We also had not verified that the legacy reference networks from RESULTS.md § 5 (a1 / b1 / c1 / e1) reproduce their old regime classification under the three-decoder protocol — the old numbers were Dec-A-only.

**What was tried (Task #25):** 10k-trial natural-HMM decode comparison of Dec A vs Dec C on R1+R2, seed 42, readout window `t∈[9,11]`, no expected/unexpected branching (the natural stream). Full agreement statistics + strata breakdowns (ambiguous vs clean; pi_low_Q1 vs pi_high_Q4; jump vs march_smooth; focused vs routine).

**What was tried (Task #23 + #24):** Legacy checkpoints a1/b1/c1/e1 re-loaded via a `MechanismType` enum shim in `src/config.py` + `torch.load(..., weights_only=False)` with `strict=False` (legacy checkpoints predate several of the current config fields). Forward pass through HMM C1 (focused + HMM cue), all three decoders applied to the same `r_l23`.

**What was tried (Task #26):** 17-row cross-decoder matrix covering: HMM C1/C2/C3/C4 on R1+R2, HMM C1 on legacy a1/b1/c1/e1, the paired-fork NEW eval on R1+R2, observational assays (M3R / HMS / HMS-T / P3P / VCD) on R1+R2, and three of those re-run with a `focused + march cue` modification. Same forward pass, three readouts.

**Outcome:**
- **Task #25.** Dec A top-1 0.5413, Dec C top-1 0.5345 on 10k natural HMM; `frac_same_pred = 0.6691`, mean circular distance 0.43 ch (≈ 2.1°). Dec A is best on the `jump` stratum (0.742 vs 0.424); Dec C is best on `ambiguous within1` (0.725 vs 0.703) and `pi_low_Q1` (0.502 vs 0.464). Both decoders cover broadly the same representational space but emphasise different strata.
- **Tasks #23+#24.** Section-5 legacy regime classifications reproduce under Dec C: a1 / b1 are weak-dampening (Δ_C −0.009 / −0.023), e1 is the best sharpener (Δ_C +0.011 — small in magnitude but sign-consistent with Dec A and Dec B), c1 is transitional (Dec A and Dec B positive, Dec C flips to −0.007 — `C outlier`). Strict all-three-decoder agreement holds on 2 of 4 rows (b1 dampening, e1 sharpening); a1 has Dec B exactly 0.0 (`B outlier` per the matrix); under the weaker A-vs-C-only check the legacy regime sign reproduces on 3 of 4 rows (a1, b1, e1).
- **Task #26.** Per-decoder profile across all 17 rows: Dec A `mean |Δ| = 0.2024 / max 0.3871`, never the sign outlier; Dec B `mean |Δ| = 0.0598`, outlier in 5 of 17 rows; Dec C `mean |Δ| = 0.0399`, outlier in 3 of 17 rows. 9 rows are all-agree. Dec A is the amplifier, Dec C is the conservative bound, Dec B is the noisiest sign-carrier.

**Decision/next step:** The claim that R1+R2 is a hybrid network — decoder-robust sharpening on paired-fork, decoder-robust dampening on observational — is now evidence-backed by the 17-row matrix. Single-decoder, single-paradigm claims in the older doc sections are deprecated in favour of this table. RESULTS.md §11 (full matrix), §13 (legacy ref), §14 (robust summary) and project_summary.md §15–§18 (mirrors) are the load-bearing sections from 2026-04-22 forward. No further training changes planned — the next step is publication-side dissection of why paradigm choice flips the sign.

**Pointers:** `results/cross_decoder_comprehensive.json` · `results/cross_decoder_comprehensive.md` · `/tmp/task25_dec_av_c_summary.json` · `RESULTS.md` §11 + §13 + §14 · `docs/project_summary.md` §15 + §17 + §18 · `ARCHITECTURE.md` § "Decoders".

---

## 2026-04-19 — Paired HMM fork paradigm × readout analysis (Tasks #19–#22)

**Context:** Task #13 (2026-04-17) showed Δ_C = +0.125 on the NEW paired-march eval for R1+R2 — a sharpening signature under Dec C. But the concurrent observational assays (matched_3row_ring, matched_hmm_ring_sequence, v2_confidence_dissection) gave **Δ_C = −0.07 to −0.06** on the same checkpoint. The sign-disagreement needed isolating: was it a paradigm difference (paired-fork constructive probe vs matched-probe observational) or a decoder artefact?

**What was tried:** A 4-condition paired HMM fork sweep (`scripts/eval_r1r2_paradigm_readout.py`): focused/routine × HMM-supplied cue/neutral zero cue = C1/C2/C3/C4. Each condition: shared pre-probe march → branch to ex/unex at the probe channel with bit-identical pre-probe state. Readout at probe-ON steps `[9, 11]`, re-centered per-trial on the true probe channel (peak at ch 18), linear-interp FWHM. Full tuning curve + peak + net + FWHM + decoder accuracies (A, B, C) recorded per branch. Task #19 added a per-trial adjacent-channel signed-offset analysis: roll to center + sign-flip by march direction, accumulate the population profile from −17 to +17 channels.

**Outcome:**
- **Decoder C Δ (ex − unex)** across 4 conditions: C1 +0.088, C2 +0.013, C3 +0.045, C4 +0.041 — all four positive. The paired-fork sharpening signature persists across every task-state × cue combination under Dec C.
- **FWHM sign across paired-fork variants.** In all 4 conditions of the 4-condition paradigm-readout sweep, expected is **wider** than unexpected (Δ_FWHM = +0.9° (C1) to +2.0° (C2)). This is the **opposite** of the NEW paired-march eval (§ 10 / Tasks #11–#13) on the same R1+R2 checkpoint, where ex FWHM (28.44°) was **narrower** than unex (29.77°): Δ_FWHM = −1.33°. Both evals use paired-fork constructive probes, so the FWHM-sign reversal is between two paired-fork variants, not between paired-fork and observational paradigms. Peak sign (ex > unex) and net-L2/3 sign (ex < unex) are consistent across both paired-fork variants; only FWHM flips.
- **Decoding-sign flip is paradigm-level.** The decoding sign flip between paired-fork (Δ_C > 0 in all 4 conditions) and matched-probe observational (Δ_C < 0 on M3R / HMS-T / VCD) is a separate, paradigm-level effect documented in RESULTS.md §11 and §12 (decoding accuracies) — it is what motivates the "hybrid" relabelling of R1+R2.
- **Adjacent-channel signed-offset curve (Task #19).** On expected trials, the `+k` flank is lower than the `−k` flank by ≈ 0.06, 0.10, 0.10 for k ∈ {1, 2, 3} under Dec C readout — a march-direction-aligned flank asymmetry. On unexpected trials, both flanks are near-symmetric. Interpretation-free observation; does not invalidate the center-peak sharpening result.
- **The sign difference between paradigms is real, not a decoder effect.** The same checkpoint, same Dec C, gives Δ_C > 0 on paired-fork (all 4 conditions) and Δ_C < 0 on matched-probe observational (all 4 assays). Paradigm choice drives the sign.

**Decision/next step:** R1+R2 is reclassified as **hybrid** (decoder-robust sharpening on paired-fork, decoder-robust dampening on matched-probe observational) in project_summary.md masthead and RESULTS.md §14. The next methodological step (see 2026-04-22 entry) was to cross-check every Δ in the doc set under all three decoders — producing the 17-row matrix.

**Pointers:** `scripts/eval_r1r2_paradigm_readout.py` · `scripts/eval_ex_vs_unex_decC_adjacent.py` · `results/r1r2_paradigm_readout.json` · `results/r1r2_paired_hmm_fork.json` · `results/eval_ex_vs_unex_decC_adjacent.json` · `RESULTS.md` §12.

---

## 2026-04-17 — Decoder C ex/unex paired eval on R1+R2 (Tasks #11–#13)

**Context:** Decoder A (the linear sensory readout frozen in Stage 2) showed a strong matched-probe Δdec(ex−unex)=+0.32 on R1+R2, but a 5-fold nearest-centroid CV (Decoder B) collapsed the gap to ≈+0.04 — within per-fold noise. Root cause: Decoder A's fixed templates, trained on the natural-march distribution, are out-of-distribution for the synthetic Pass B compound bumps used in matched-probe-3pass. A new decoder was needed for the expectation-suppression analysis.

**What was tried:** **Decoder C** — a standalone `Linear(36, 36)` trained on 100k synthetic orientation-bump patterns (50k single-orientation σ=3 ch with amplitudes ∈ [0.1, 2.0]; 50k multi-orientation K∈{2,3} with strictly-max amplitude as the label; Gaussian noise σ=0.02; Adam lr=1e-3, batch 256, ≤30 epochs, early-stop patience 3, seed 42). Held-out synthetic accuracy 0.81 (single 0.98 / multi 0.65); real-network natural-HMM R1+R2 accuracy 0.66 non-amb / 0.53 all. Then a **paired ex/unex eval** (Tasks #12/#13) was run on R1+R2 simple_dual emergent_seed42: 12 N values (4..15) × 200 trials/N = 2400 paired ex/unex trials. Random S ∈ [0°, 180°), D ∈ [25°, 90°], CW/CCW 50/50 per trial, fixed `task_state=[1,0]` focused, cue at expected-next orientation in BOTH branches, contrast 1.0. Per-trial RNG seed = `42 + trial_idx` (independent of N → bit-identical pre-probe march across N). Readout: probe-ON window steps [9:11] mean-pooled, then per-trial roll-to-center on the true probe channel (peak at ch18) with linear-interp FWHM (same convention as `ce1b34e` / `matched_hmm_ring_sequence.py`).

**Outcome:** Pooled across N (n=2400 paired trials):

| Metric | Expected | Unexpected | Δ (ex − unex) |
|---|---:|---:|---:|
| Decoder C accuracy | 0.707 ± 0.009 | 0.581 ± 0.010 | +0.125 |
| Net L2/3 (sum 36 ch) | 4.99 ± 0.01 | 6.13 ± 0.02 | −1.15 |
| Peak at true-ch | 0.773 ± 0.003 | 0.626 ± 0.004 | +0.147 |
| FWHM | 28.4° ± 0.10 | 29.8° ± 0.19 | −1.33° |

All four signs hold at every N from 4 to 15. Pre-probe state bit-identical across branches (max|ex−unex|=0.00). All 2400 trials produced valid FWHM crossings in both branches. Expected trials show **lower net L2/3 activity, higher peak at the stimulus channel, narrower tuning, and higher decoding accuracy** than unexpected trials.

**Interpretation framing (literal).** Under the project's operational dampening definition (lower activity AND lower decoding on expected), this pattern passes on net L2/3 but fails on decoding, peak, and FWHM. Under the Kok 2012 sharpening definition (narrower tuning, higher peak, better decoding, lower total activity), this pattern matches. Under Richter preserved-shape dampening (lower peak, preserved FWHM, preserved decoding), this pattern does not match — the peak goes the wrong direction. **R1+R2 is set as the canonical default checkpoint** for expectation-suppression analyses on the `dampening-analysis` branch from this point forward. Decoder C is the preferred decoder; Decoder A is retained only as a reference due to its OOD/SNR-confound behavior on synthetic probes — Network_mm / Network_both / HMM Expected-vs-Unexpected numbers that used Decoder A are decoder-dependent and should be re-checked under Decoder C before publication.

**Pointers:** `scripts/eval_ex_vs_unex_decC.py` · `results/eval_ex_vs_unex_decC.json` · `scripts/train_decoder_c.py` · `checkpoints/decoder_c.pt` · `docs/figures/eval_ex_vs_unex_decC.png` · `RESULTS.md` § 10 · `docs/rescues_1_to_4_summary.md` § "Decoder A artefact note (2026-04-17)".

---

## 2026-04-13 — Re-centered tuning analysis: R4 resembles Richter preserved-shape dampening (`ce1b34e`, `60bb69b`)

**Context:** The original `expected vs unexpected` tuning analysis indexed `r_l23` by physical channel and averaged within (regime × bucket) buckets. Because each bucket pools trials at many true orientations, the average smears the population peak across channels — the bucket-mean curve inflates apparent FWHM and can hide preserved-shape dampening.

**What was tried:** For each trial, roll `r_l23[t=9]` so the true-stimulus channel lands at the array midpoint (`np.roll(..., shift=18 - true_ch)`), then average within buckets. Replace bin-counted FWHM with linear interpolation at half-max crossings.

**Outcome:** Re-centered analysis on all 4 checkpoints (Relevant task_state):
- Baseline: Δ total +3.5%, Δ peak +0.5%, Δ FWHM +1.1° → no modulation.
- R1+R2: Δ total +18%, Δ peak +13%, Δ FWHM +1.5° → preserved-shape dampening.
- R3: Δ total +15%, Δ peak +0.5%, Δ FWHM +1.1° → total-only (divisive) dampening.
- **R4: Δ total +19%, Δ peak +15%, Δ FWHM +1.5° → cleanest Richter-like preserved-shape dampening.**

**Decision/next step:** Corrects the earlier "subtractive predictive coding" interpretation. Dampening is task-state-invariant though, so the preregistered BOTH-regime criterion remains unmet. Open paths: (A) write up as negative result on the dissociation hypothesis, (B/C) further architectural changes to recover dissociation.

**Pointers:** `docs/rescues_1_to_4_summary.md` § "Update (2026-04-13)" · `RESULTS.md` § 9 · `docs/figures/tuning_ring_recentered_{baseline,r1_2,r3,r4}.png` · scripts `plot_tuning_ring_extended.py`, `plot_tuning_ring_heatmap.py`, `plot_tuning_exp_vs_unexp.py`.

---

## 2026-04-13 — Rescue 5: shape-matched predictive suppression (`39375ac`)

**Context:** R1–R4 all showed expected < unexpected on BOTH activity AND decoding, with expected FWHM (then-measured non-re-centered) appearing broader. Hypothesis: the near-one-hot `q_pred` over-subtracts the peak of `r_l23` ("central clipping"). A shape-matched bump in the same feature basis as `r_l23` should preserve peak while still suppressing predicted orientations.

**What was tried:** Calibrate a fixed buffer `T_stage1 ∈ R^{N×N}` once at the Stage 1 → Stage 2 boundary (row j = mean Stage-1 L2/3 response at orientation j with FB=0). At loss time, `q_match = q_pred @ T_stage1` replaces the raw softmax `q_pred` inside `expected_suppress_loss`. Single delta on top of R1+R2 (no VIP, no DeepTemplate).

**Outcome:** R5 did NOT reach Richter regime. Peak clipping largely unchanged (~35.5% peak gap). FWHM gap closed only modestly (~7° → ~1.76°). Calibration verified (`T_stage1` row_peak_mean=0.113, row_sum_mean=0.628).

**Decision/next step:** R5 shelved. Subsequent re-centered analysis (next entry, chronologically) showed the original "central clipping" interpretation was itself an analysis artefact — R4's per-trial response is preserved-shape dampening, not the central-clipping subtractive pattern R5 was designed to fix.

**Pointers:** `config/sweep/sweep_rescue_5.yaml` · `docs/rescues_1_to_4_summary.md` (R5 section).

---

## 2026-04-13 — Rescue 4: DeepTemplate + error-based mismatch readout (`564ae1b`)

**Context:** R1+R2 added precision gating + feature-specific suppression but kept the shared-decoder bottleneck (sensory and mismatch heads both read `r_l23`). External review (gap 5) flagged this as forcing the network to balance two objectives on one representation.

**What was tried:** Add a `DeepTemplate` leaky integrator (τ=10, learnable scalar gain) maintaining a V1-side expectation template `r_template = gain · q_pred · pi_pred_eff`. Compute `r_error = relu(r_l23 - r_template)` per timestep. The mismatch head reads from `r_error`; the sensory decoder still reads `r_l23`. Stacked on R1+R2 (no VIP).

**Outcome:** Δ_sens = +0.241, Δ_mm = +0.168, +34% L2/3 total expectation suppression. mm dissociation regressed from R1+R2's +0.220 — the ReLU on `r_error` clips half the signal (template-over-predicts case). `Δ_sens` essentially unchanged. M7 δ=10° = +0.315.

**Decision/next step:** R4 has the strongest expectation suppression but did not produce the BOTH-regime dissociation. Triggered R5 (shape-matched suppression) attempting to fix the apparent central clipping.

**Pointers:** `config/sweep/sweep_rescue_4.yaml` · `docs/rescues_1_to_4_summary.md` § "Rescue 4" · `tests/test_network.py::TestRescue4DeepTemplate`.

---

## 2026-04-12 — Rescue 3: VIP-SOM disinhibition with structured surround (`b76001b`)

**Context:** R1+R2 added precision gating but retained the simple feedback E/I split with no disinhibitory pathway (gap 2). Hypothesis: a VIP→SOM disinhibition with a structured center-surround kernel could let feedback selectively suppress flanks while preserving the predicted-channel peak.

**What was tried:** Add a `VIPRing` leaky integrator population (τ=10) driven by a new `head_vip = Linear(16, 36)` from V2; a softplus-positive learnable weight `w_vip_som` connects VIP onto SOM (`effective_som_drive = relu(som_drive_fb - w_vip * r_vip)`). SOM drive is filtered by a circular Gaussian kernel (σ=20°). Stacked on R1+R2.

**Outcome:** Δ_sens = +0.210, Δ_mm = +0.155, +19% L2/3 total expectation suppression (intermediate between baseline and R1+R2/R4). FWHM ON narrowed to 17.92° (strongest narrowing of the rescue chain, FWHM Δ = −8.76°). mm dissociation regressed from R1+R2.

**Decision/next step:** R3 produced the strongest FB-on-vs-off sharpening but no peak-at-true differentiation between expected and unexpected. Re-centered analysis later showed R3 produces total-only (divisive) dampening — distinct from R4's preserved-shape pattern. Triggered R4 (laminar separation via DeepTemplate).

**Pointers:** `config/sweep/sweep_rescue_3.yaml` · `docs/rescues_1_to_4_summary.md` § "Rescue 3" · `docs/figures/tuning_ring_recentered_r3.png`.

---

## 2026-04-12 — Rescue 1+2: precision gating + feature-specific expected_suppress (`761a5e8`)

**Context:** External review identified 5 architectural gaps (next entry). Gaps 1 (precision as bookkeeping) and 4 (global losses) were the smallest changes — addressing them first establishes whether feedback can be made precision-sensitive and whether suppression can be made feature-specific.

**What was tried:** Two minimal changes composed into one sweep. (1) `scaled_fb = feedback_signal * feedback_scale * (pi_pred_raw / pi_max)` — feedback amplitude is gated by the network's own precision estimate. (2) `expected_suppress_loss` switched from `mean(|r_l23|)` to `(r_l23_windows * q_pred_windows).sum(-1)` — penalize only the q_pred-aligned component of L2/3 activity. Restored `lambda_state = 1.0` for a coherent task-state prior.

**Outcome:** Δ_sens = +0.248, **Δ_mm = +0.220** (strongest mm dissociation of the entire rescue chain). +33% L2/3 total expectation suppression, feedback-driven (FB-OFF gap is −0.10). M7 δ=10° = +0.311, FWHM ON 21.59° (FWHM Δ = −5.09°).

**Decision/next step:** R1+R2 became the substrate for R3 (added VIP) and R4 (added DeepTemplate) — both attempted to recover the BOTH-regime dissociation that R1+R2 alone did not produce.

**Pointers:** `config/sweep/sweep_rescue_1_2.yaml` · `docs/rescues_1_to_4_summary.md` § "Rescue 1+2" · `docs/figures/tuning_ring_recentered_r1_2.png`.

---

## 2026-04-12 — External code review: 5 architectural gaps identified (`4632849`)

**Context:** Network_mm and Network_both both showed near-zero Δ_sens with Δ_mm only attainable by sacrificing sensory dissociation (Fix 2 regressed mm). The single-network-dual-regime approach as instantiated could not produce both regime-selective sharpening AND dampening from one architecture. External review was sought to identify whether the failure was loss-side, training-side, or architectural.

**What was tried:** External review of the failed dual-regime experiments captured in commit `4632849` ("Failed dual-regime experiments: comprehensive summary + all code"). Identified 5 architectural gaps in the existing single-network instantiation.

**Outcome:** Five gaps documented:
1. Precision as bookkeeping (computed but not used to modulate feedback amplitude).
2. No disinhibitory mechanism (VIP stripped in `1987fab`; feedback had only raw E/I split).
3. No laminar separation of expectation from evidence (no dedicated template population).
4. Global, not feature-specific, suppression losses.
5. Shared-decoder bottleneck (sensory and mismatch read same feature).

**Decision/next step:** Four rescues (R1+R2, R3, R4, R5) designed to address one or more gaps incrementally. Branch `failed-dual-regime-experiments` cut from `single-network-dual-regime` HEAD.

**Pointers:** `docs/rescues_1_to_4_summary.md` § "The 5 architectural gaps (external review)".

---

## 2026-04-11 — Network_both: per-regime head_feedback (Fix A + Fix 2) (`fb7a1a1`)

**Context:** Network_mm (previous entry, chronologically) achieved Δ_mm = +0.212 but Δ_sens was flat. Hypothesis: the shared single `head_feedback(16→36)` produces a broadband gain at the predicted orientation in both regimes — the network has no per-regime degree of freedom. Splitting into two heads, gated by task_state, could let each regime specialize.

**What was tried:** Two `Linear(16, 36)` heads — `head_feedback_focused` and `head_feedback_routine` — gated by `task_state[0:1]` and `task_state[1:2]` respectively. Single shared V2 GRU upstream. `use_per_regime_feedback=True`. Otherwise identical to Network_mm.

**Outcome:** Δ_sens = +0.052 (weak; oscillated ±0.05 around zero), **Δ_mm regressed to +0.140** (from Fix-A's +0.212). Reallocated signal from mismatch to partial sensory without expanding the dissociation budget. M7 focused +0.342 / routine +0.320; FWHM focused 22.87° / routine 19.44°.

**Decision/next step:** Per-regime split in feedback alone is insufficient. The shared L4→PV→L2/3→SOM substrate is the bottleneck, not the feedback head. Triggered the external code review (next entry, chronologically).

**Pointers:** `config/sweep/sweep_simple_dual_fix2.yaml` · `docs/project_summary.md` § 5.

---

## 2026-04-11 — Network_mm: MLP mismatch_head (Fix A) (`9d1e954`, `403bf25`)

**Context:** The single-network-dual-regime baseline (`194a3c7`) was running but the linear mismatch head was plateauing at val_acc ~0.74. Debugger investigation (Task #6) showed an MLP could reach 0.89 on the same input.

**What was tried:** Replace `mismatch_head = Linear(36, 1)` with `Linear(36,64) → ReLU → Linear(64,1)`. Add `loss_fn` state to checkpoints so mismatch_head weights survive checkpoint reload. Single shared V2 (no per-regime split). Otherwise identical baseline.

**Outcome:** **Δ_mm = +0.212** (mm_acc_irr = 0.906, mm_acc_rel = 0.694) — strong mismatch-only dissociation. **Δ_sens = +0.001** (s_acc_rel = 0.652, s_acc_irr = 0.651) — flat. M7 focused +0.334, routine +0.329. M10 focused 5.169, routine 4.776 (both regimes amplify; no dampening).

**Decision/next step:** Mismatch dissociation works with the right head capacity; sensory dissociation does not emerge with a shared feedback head. Triggered Network_both (per-regime feedback heads).

**Pointers:** `config/sweep/sweep_simple_dual.yaml` · `docs/project_summary.md` § 5.

---

## 2026-04-11 — Phase 2.4.1: alpha_net causal E/I gate confirmed after freeze-bug fix (`ef4f102`, `cde523f`)

**Context:** Phase 2 (`4ae515d`) added a learnable causal E/I gate `alpha_net = Linear(3, 2)` taking `(task_state[0:2], pi_pred_raw)` to per-regime gains `(g_E, g_I)` multiplying `center_exc` and `som_drive_fb`. Initial training showed the gate stuck at identity (Δg_E ≈ +0.004) — suspected gradient was reaching `alpha_net` but the optimizer was not updating it.

**What was tried:** Debugger localized the failure to `stage1_sensory.py` blanket-freezing all parameters before Stage 1 and `unfreeze_stage2()` not re-enabling `alpha_net`. `create_stage2_optimizer()` filtered by `requires_grad`, silently dropping the frozen `alpha_net` group. Fix in `cde523f`: explicit `alpha_net.requires_grad_(True)` in `unfreeze_stage2`. Phase 2.4 (`ef4f102`) added a routine E/I symmetry-break loss + configurable alpha_net LR multiplier on top.

**Outcome:** After fix, alpha_net learned focused/routine differentiation in the feedback E/I split: **Δg_E = +1.34, Δg_I = −1.60**. All 4 preregistered Phase-2 gates passed. Result is writeup-ready (independent pathway, not on R-series branch).

**Decision/next step:** Alpha_net gate as a scalar E/I modulator confirmed but did not generalize to per-channel selectivity. Phase 2.4.2 (`2077ac7`) softened hyperparameters after post-fix runaway. Architectural pathway forked from R-series: alpha_net + R-series merge identified as a candidate Rescue 5+ direction (not yet attempted).

**Pointers:** `~/.claude/projects/.../memory/phase2_gate_result.md` · `docs/project_summary.md` § 4 · commits `4ae515d`, `ef4f102`, `cde523f`, `2077ac7`.

---

## 2026-04-11 — Baseline: simple-dual-regime architecture established (`194a3c7`)

**Context:** Phase 2.x alpha_net arc converged on the conclusion that a scalar E/I gate could not produce per-channel selectivity. After legacy code removal (`1987fab`, 2026-04-10: VIP, apical gain, basis functions, coincidence gate, MechanismType all stripped), a clean restart was needed: one network, three loss terms, Markov task_state, observe four accuracies (`s_acc_rel`, `s_acc_irr`, `mm_acc_rel`, `mm_acc_irr`).

**What was tried:** Establish the simple-dual-regime baseline in commit `194a3c7`: single shared V2, single shared `head_feedback`, Markov task_state with `p_switch=0.2`, per-presentation loss gating:
```
loss = task_state[0] * (3.0*sensory + 0.0*mismatch + 1.0*energy)
     + task_state[1] * (0.3*sensory + 1.0*mismatch + 1.0*energy)
```

**Outcome:** Baseline trained successfully — provides the substrate for all subsequent rescue experiments. End-state metrics (later measured): s_acc_rel=0.696, s_acc_irr=0.673 (Δ_sens +0.023), mm_acc_rel=0.462, mm_acc_irr=0.533 (Δ_mm +0.071). No regime dissociation yet — both accuracies near-tied.

**Decision/next step:** Triggered the Fix A (MLP mismatch_head) and Fix 2 (per-regime feedback) iterations. Baseline checkpoint became the starting point for all rescues R1+R2, R3, R4, R5.

**Pointers:** `config/sweep/sweep_simple_dual.yaml` · `~/.claude/projects/.../memory/target_experimental_setup.md` · `docs/project_summary.md` § 3 · `RESULTS.md` (3-regime predecessor sweep).
