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

## 2026-04-24 — Decoder D (FB-ON neutral, 2 variants) + Decoder E (Dec-A-spec, post-Stage-2) per-ckpt + 7-column matrix + Dec A vs Dec E dissociation

**Context:** The 2026-04-22 matrix had three decoders (A/B/C). 2026-04-23 added Dec A′ (R1+R2 only) as a stable-target sanity check on Dec A. This entry adds two further axes: Dec D — a neutral FB-ON localizer trained on each frozen fully-trained network with a balanced paired-fork ex+unex dataset — and Dec E — a Dec-A-spec retrain that uses the natural HMM stream's own stochastic task_state (NOT 50/50 pinned). Both are per-ckpt across all 5 networks (R1+R2 + a1 / b1 / c1 / e1). The 17-row matrix now has 7 decoder columns.

**What was tried (Dec D, Task #4).** Paired-fork design: N_pre ∈ U{4..10} pre-probe march at 5°/step, probe rendered at target_ch; unex branch offsets march_end_ch by ±U[5,18] channels (25°–90° signed rotation). Cue at expected-next (same for ex and unex), focused task_state, contrast U[0.4, 1.0], feedback_scale = 1.0 throughout (normal feedback). 900 train + 100 val per (target_ch × branch) cell → 64 800 train + 7 200 val = 72 000 samples per ckpt. Two variants: D-raw (Linear(36, 36)+bias on raw `r_l23`) and D-shape (on `r_l23 / (r_l23.sum(1) + 1e-8)`). Adam lr=1e-3 wd=1e-4, CE, early-stop patience 3, max 30 epochs, seed 42. Training script: `scripts/train_decoder_d_fbON_neutral.py`. Important history: an earlier FB-off variant was invalidated mid-task (the user flagged that FB=0 during training means D never sees the FB-on manifold used at eval), undone at commit `46033fb` before push, and retrained with FB on; the `_fbON_neutral_` naming distinguishes the valid artefacts.

**What was tried (Dec E, Task #5).** Same arch as Dec A (Linear(36, 36)+bias), same lr=1e-3 Adam no weight decay, seed 42, 5000 gradient steps, val pool seed 1234, readout `t∈[9,11]`. Key difference from Dec A′: task_state follows the HMM generator's own distribution (Markov `task_p_switch=0.2` per R1+R2 yaml; Bernoulli-per-batch for legacy configs that don't set `task_p_switch`). Per-checkpoint training. Training script: `scripts/train_decoder_e.py`. A post-training Dec A comparison bug (legacy ckpts lack `ckpt['loss_heads']`) crashed the a1 / b1 / c1 runs after completing 5000 steps; crash-safety snapshots at step 4000 were promoted to final ckpts via `/tmp/_recover_decE.py`, and the trainer was patched (save-before-compare + decoder_state fallback). Val_acc deltas step 4000 → step 5000 on the comparable R1+R2 and e1 runs were ≤ 0.02, so step-4000 recovery is within ~2 pp of step-5000 projection; all three ckpts are tagged `early_terminated_at_step_4000_due_to_crash` in their training JSONs.

**What was tried (Matrix rerun).** Dec D / Dec E plumbed through `scripts/cross_decoder_eval.py` and `scripts/r1r2_paradigm_readout.py` as optional `--decoder-d-{raw,shape}-path` args. Dec E used the ckpt-patch trick (`scripts/_make_decE_ckpt.py`) — patch each net's ckpt with its own Dec E in the `orientation_decoder` / `decoder_state` slots, re-run the 3 R1+R2 pipelines + 4 legacy HMM C1 runs, relabel the output's `decA_delta` as `decE_delta`. Merger `scripts/merge_decE_matrix.py` combines the base 2026-04-24 Dec D matrix (`results/cross_decoder_comprehensive_withD_fbON.json`), the 2026-04-23 Dec A′ matrix, and the Dec E rerun outputs into the final 7-column `results/cross_decoder_comprehensive_with_all_decoders.{json,md}`. Each row has Δ_A / Δ_A′ / Δ_B / Δ_C / Δ_D-raw / Δ_D-shape / Δ_E; ABC sign-agreement flags preserved.

**Outcome:**
- **Per-net 10k natural HMM top-1 (seed 42).**
  ```
  net     D-raw    D-shape   A        A'       C        E
  r1r2    0.3634   0.3726    0.5413   0.5486   0.5345   0.5467
  a1      0.4130   0.5396    0.5907   —        0.5050   0.3542
  b1      0.4080   0.5652    0.5830   —        0.5028   0.3507
  c1      0.3848   0.3597    0.4476   —        0.4510   0.4257
  e1      0.4074   0.3995    0.4887   —        0.4647   0.4778
  ```
  Dec D-raw / Dec D-shape are OOD for the 10k HMM eval (trained focused-only paired-fork; tested on 50/50 focused/routine + ambiguous) so their 10k numbers are not apples-to-apples with A/A′/C/E.
- **Dec A vs Dec E dissociation on dampening legacy nets.** Dec E matches Dec A to within ±0.5 pp on R1+R2 and e1, to within ±2.2 pp on c1, but **Dec A ≫ Dec E by 23.6 pp on a1 and 23.2 pp on b1**. On the 17-row matrix Dec E flips sign vs Dec A on exactly those two rows (a1 + b1 HMM C1): Δ_A = −0.031 / −0.033 → Δ_E = +0.040 / +0.024. On these rows Dec E is the **lone positive** across all seven decoders (A, A′, B, C, D-raw, D-shape, E) — all six others report Δ < 0. The implication is that on dampening legacy configs, Dec A captures a representational structure that 5000 steps of fresh post-Stage-2 natural-HMM training cannot reproduce, and the Δ_A sign on these rows is training-regime-dependent.
- **Dec D Kok-style signature on row 12.** HMS-T native on R1+R2 gives Δ_A = −0.303, Δ_B = −0.143, Δ_C = −0.078, Δ_D-raw = −0.053, Δ_E = −0.207 (five-decoder all-agree dampening) — but **Δ_D-shape = +0.166**. D-shape's shape-only input reports ex > unex on the same `r_l23` where every amplitude-sensitive decoder reports ex < unex. This is the Kok-framework "expectation suppresses net amplitude while sharpening the orientation-pattern shape" co-occurrence; no other row of the 17-row matrix shows the divergence at material magnitude.
- **Per-decoder magnitude profile (2026-04-24).** A 0.2056, A′ 0.1902 (13 R1+R2 rows), B 0.0485, C 0.0416, D-raw 0.0520, D-shape 0.0585, E 0.1934. 11 of 17 rows are ABC all-agree (up from 9 in 2026-04-22; Δ_B run-to-run drift reshuffled 2 borderline rows — HMM C3 gained ALL-agree, HMM C1 on a1 legacy gained ALL-agree).
- **e1 re-classification.** The 2026-04-22 "decoder-robust sharpening" label on e1 HMM C1 relied on Δ_C = +0.011 which crossed zero to −0.002 in the 2026-04-24 rerun (within ±0.03 CPU FP drift envelope). Under Dec D / Dec E split further: A / A′ (none for legacy) / B / E all positive; C / D-raw / D-shape near zero. e1 is **training-regime-dependent**, not decoder-robust.

**Decision/next step:** The 2026-04-22 hybrid R1+R2 finding (sharpening on paired-fork, dampening on observational) holds on all 13 R1+R2 rows under the 7-column matrix. The new axes reveal two new phenomena: (1) a Dec A-vs-Dec E dissociation on the dampening legacy networks (a1, b1) which contests the robustness of Δ_A sign on those rows, and (2) a Dec D-shape sharpening signature on HMS-T native that diverges from every amplitude-sensitive decoder — the only row in the matrix with this property. e1 is re-flagged as training-regime-dependent. No further training changes are planned for Task #4 / Task #5. Docs updated in this pass (Task #6).

**Pointers:** `results/cross_decoder_comprehensive_with_all_decoders.{json,md}` · `results/cross_decoder_comprehensive_withD_fbON.{json,md}` · `results/decoder_d_fbON_neutral_{net}.json` · `results/decoder_d_fbON_all_eval.json` · `results/decoder_e_training_{net}.json` · `results/decoder_e_stratified_eval_{net}.json` · `checkpoints/decoder_d_fbON_neutral_{raw,shape}_{net}.pt` · `checkpoints/decoder_e_{net}.pt` · `scripts/train_decoder_d_fbON_neutral.py` · `scripts/train_decoder_e.py` · `scripts/eval_decoder_d_on_hmm.py` · `scripts/eval_decoder_e_stratified.py` · `scripts/_make_decE_ckpt.py` · `scripts/merge_decE_matrix.py` · `ARCHITECTURE.md` § "Decoders" (6-row taxonomy + Kok-signature flag) · `RESULTS.md` §11 (7-column matrix + Dec A→E flips) + §14 (e1 reclassification) · `docs/project_summary.md` §15 + §18.

---

## 2026-04-23 — Stable-target Dec A′ retrain + 13-row matrix rerun (Task #1 Dec A′ line)

**Context:** The user flagged that Dec A (`loss_fn.orientation_decoder` in `src/training/stage1_sensory.py:127-163`) is trained jointly with L2/3+PV from step 0 of Stage 1, so its weights fit a moving target — early-training L2/3 ≠ late-training L2/3. Question: does the moving-target training contaminate the Task #26 17-row dampening-vs-sharpening matrix?

**What was tried (Part A):** A fresh `Linear(36, 36)` with bias was trained for 5000 Adam steps (lr=1e-3) on `r_l23` streamed through the **fully-trained, frozen** R1+R2 emergent_seed42 network. Net.eval() + `requires_grad_(False)` on every net param, verified by assertion at setup. Per step: batch 32 × seq_length 25 = 800 readouts at `t∈[9,11]`, ambiguous kept in, 50/50 focused/routine task_state. Seed 42 for init + training stream; val pool seed 1234 (~8k readouts, eval every 500 steps). Saved as `checkpoints/decoder_a_prime.pt`. Training driver: `scripts/train_decoder_a_prime.py`. Stratified eval (Task #25-identical 10k natural HMM design, same strata — ambiguous/clean, pi_low_Q1/pi_high_Q4, low_pred_err_le5deg/high_pred_err_gt20deg, focused/routine, march_smooth/jump): `scripts/eval_decoder_a_prime_stratified.py`.

**What was tried (Part B):** Patched the R1+R2 ckpt in memory (`loss_heads.orientation_decoder` + `decoder_state` replaced with Dec A′ weights; network parameters untouched) via `scripts/_make_decAprime_ckpt.py` → `/tmp/r1r2_ckpt_decAprime.pt`. Reran the 3 R1+R2 pipelines (`r1r2_paradigm_readout.py`, `cross_decoder_eval.py` native + modified) against that patched ckpt on CPU (seed 42, identical args to the original 2026-04-22 invocations). Legacy a1/b1/c1/e1 rows retained their own stored Dec A per lead directive (Dec A′ was trained on R1+R2 L2/3, applying it to legacy L2/3 would be a transfer test not the same comparison). Re-aggregated: `scripts/aggregate_cross_decoder_matrix.py` → `results/cross_decoder_comprehensive_decAprime.{json,md}`. Row-by-row diff: `scripts/diff_decAprime_matrix.py` → `results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

**Outcome:**
- **Dec A′ training.** Final val-pool top-1 = 0.5560 (vs Dec A 0.5473 on the same held-out pool). Training curve monotonic: 0.032 (step 0) → 0.352 (500) → 0.481 (2k) → 0.550 (4k) → **0.556 (5k)**.
- **Stratified eval on 10k natural HMM.** Dec A 0.5413 / Dec A′ **0.5486** / Dec C 0.5345 (top-1); Dec A′ MAE_ch 0.790 (vs Dec A 0.820, Dec C 0.862). `frac_same_pred(A, A′) = 0.8200`; `frac_same_pred(A, C) = 0.6691`; `frac_same_pred(A′, C) = 0.6367`. Dec A′ edges Dec A on clean / high_pred_err / pi_high_Q4 (+1.6 / +2.5 / +2.6 pp) and slightly deficits on ambiguous / low_pred_err / pi_low_Q1 (−1.4 / −3.9 / −2.9 pp).
- **13-row R1+R2 matrix swap.** Zero Δ-sign flips in Δ_A → Δ_A′. `|Δ_A′ − Δ_A| ≤ 0.094` (median 0.025, mean 0.032). The three largest shifts are on sharpening-side rows — HMS-T native +0.081, HMS-T modified +0.094, M3R native +0.061 — all toward smaller |Δ|. Holding Δ_B / Δ_C fixed at the original-run values, **zero rows change sign-agreement class** under the Dec A → Dec A′ swap. The Dec A′ rerun's Δ_B / Δ_C values drifted by ≤ 0.03 from the original matrix (same seed, same CPU, residual FP run-to-run noise); under those run-matched Δ_B / Δ_C, two rows shift class — HMM C3 tightens to ALL-agree (run Δ_B = +0.024 vs original −0.007), M3R native loosens to B-outlier (run Δ_B = +0.003 vs original −0.008). Both are driven by Δ_B noise ±0.03, not the Dec A → Dec A′ swap.

**Decision/next step:** The moving-target concern on Dec A does not materially change the 13-row dampening-vs-sharpening pattern — the robust-findings categories in RESULTS.md §14 are unchanged. Dec A′ is now documented as the stable-target sanity check in ARCHITECTURE.md § "Decoders" (4-row taxonomy A / A′ / B / C). No matrix claims need revision. Per-legacy Dec A′ retrain (option (iii)) was parked as a possible follow-up for cross-network symmetry; not required by the present result.

**Pointers:** `checkpoints/decoder_a_prime.pt` · `results/decoder_a_prime_training.json` · `results/decoder_a_prime_stratified_eval.json` · `results/cross_decoder_comprehensive_decAprime.{json,md}` · `results/cross_decoder_comprehensive_decAprime_diff.{json,md}` · `scripts/train_decoder_a_prime.py` · `scripts/eval_decoder_a_prime_stratified.py` · `scripts/_make_decAprime_ckpt.py` · `scripts/diff_decAprime_matrix.py` · `ARCHITECTURE.md` § "Decoders" + § "Stable-target decoder sanity check" · `RESULTS.md` § 11 ("Dec A → Dec A′ swap summary") + § 14 · `docs/project_summary.md` § 15.

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
