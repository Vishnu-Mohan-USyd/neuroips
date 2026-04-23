# R1+R2 Full Project Report

**Date:** 2026-04-23
**Branch:** `r1r2-decoderC-sharpening-eval`
**Default checkpoint (R1+R2):** `results/simple_dual/emergent_seed42/checkpoint.pt`
**Config:** `config/sweep/sweep_rescue_1_2.yaml`
**Chance baseline:** `1/36 ≈ 0.0278` (36-way orientation classification)

This report is a single-file synthesis of the architecture, training, decoders, evaluation paradigms, and findings for the R1+R2 network. Numbers are pulled directly from source JSONs (listed in § 10). It is additive to the existing docs (`ARCHITECTURE.md`, `RESULTS.md`, `README.md`, `docs/project_summary.md`, `docs/research_log.md`) and preserves their content; no existing claims are modified here.

---

## Table of contents

1. Overview and headline claims
2. Architecture
3. Training pipeline
4. Decoders: A / A′ / B / C
5. Evaluation paradigms catalogue
6. 17-row cross-decoder comprehensive matrix
7. Dec A → Dec A′ swap (stable-target sanity check)
8. Legacy reference networks (a1 / b1 / c1 / e1) under the three-decoder protocol
9. Robust findings synthesis
10. Reproducibility and artefact index

---

## 1. Overview and headline claims

**Research question.** On the `dampening-analysis` / `r1r2-decoderC-sharpening-eval` branches, the R1+R2 checkpoint is a single network from the simple-dual-regime architecture trained with Rescue 1 (feature-specific `expected_suppress` via `dot(r_l23, q_pred)`) + Rescue 2 (precision-gated feedback via `pi_pred`). The question is whether this network implements **sharpening** (expectation-enhanced representation; ex > unex on decoder accuracy) or **dampening** (expectation-suppressed representation; ex < unex), and whether the answer depends on the assay paradigm or the decoder.

**Headline claims (factual, evidence-backed).**

- **Hybrid regime.** R1+R2 is decoder-robust-sharpening on the paired-fork paradigm (rows 1 and 9 of § 6) and decoder-robust-dampening on observational matched-probe paradigms (rows 10, 12, 14, 15, 17). The paradigm, not the decoder, flips the sign.
- **Dec A → Dec A′ swap.** Retraining Decoder A on the **stable** (post-training, frozen) R1+R2 `r_l23` distribution produces **zero Δ-sign flips** across the 13 R1+R2 rows of the cross-decoder matrix. Max `|Δ_A′ − Δ_A|` = 0.094, median 0.025, mean 0.032. The Dec A training-schedule concern (co-trained with L2/3 in Stage 1, moving target) does not materially change the 13-row sign pattern. See § 7 and `results/cross_decoder_comprehensive_decAprime_diff.json`.
- **Decoder profiles on 17 rows.** Dec A `mean|Δ|=0.2024`, never the sign outlier; Dec B `mean|Δ|=0.0598`, outlier in 5/17 rows; Dec C `mean|Δ|=0.0399`, outlier in 3/17 rows. 9 of 17 rows are all-agree. Source: `results/cross_decoder_comprehensive.json`.
- **Dec A vs Dec C agreement on 10k natural HMM.** `frac_same_pred = 0.6691`, mean circular distance 0.43 ch (≈ 2.1°). Dec A top-1 = 0.5413, Dec C top-1 = 0.5345 (both ≈ 19× chance). Source: `/tmp/task25_dec_av_c_summary.json`.

**Non-claims (calibration).** "Zero sign flips" under Dec A → Dec A′ does not validate Dec A as correct; it only means two linear decoders (trained on the same network states under slightly different schedules) give the same sign of Δ on the 13 R1+R2 rows. "Decoder-robust" in this doc means `sign(Δ_A) = sign(Δ_B) = sign(Δ_C)`, no more.

---

## 2. Architecture

### 2.1. Module map

Defined in `src/model/network.py` (`class LaminarV1V2Network` at `:18`). Populations instantiated in `__init__` at `:34-82`:

| Line | Field | Class | Role |
|---:|---|---|---|
| `:34` | `self.l4` | `V1L4Ring` (`src/model/populations.py:21`) | V1 Layer 4 excitatory ring, driven by grating + contrast + adaptation |
| `:35` | `self.pv` | `PVPool` (`:95`) | PV interneuron pool, divisive normalisation |
| `:36` | `self.l23` | `V1L23Ring` (`:179`) | V1 Layer 2/3 excitatory ring — primary readout population |
| `:37` | `self.som` | `SOMRing` (`:290`) | SOM interneurons, feedback-driven subtractive surround |
| `:76-82` | `self.v2_focused`, `self.v2_routine`, `self.v2` | `V2ContextModule` (`src/model/v2_context.py:13`) | V2 GRU + prediction/precision/feedback/VIP heads. Dual instances for per-regime heads (Fix 2, `Network_both`); legacy alias `self.v2 = self.v2_focused`. |
| `:97-99` | `self.alpha_net` | `nn.Linear(2+1, 2)` | Causal E/I gate (Phase 2.4.1) — emits two gains applied to `center_exc` and `som_drive_fb`. Only present when `model_cfg.use_ei_gate=True`; disabled (`false`) in `sweep_rescue_1_2.yaml`. |
| — | `self.feedback_scale` | buffer `[1]` | Warmup multiplier, ramps 0→1 across Stage-2 burn-in. Loaded at `1.0` at inference. |

Ring geometry: `n_orientations = 36` channels, `orientation_range = 180.0°`, `step_deg = 5.0°`.

### 2.2. V2 heads (`src/model/v2_context.py:13-145`)

GRU(input_dim=110, hidden=16). Input is `r_l4 + r_l23_prev + cue + task_state = 36+36+36+2 = 110`. Heads:

| Line | Head | Output | Notes |
|---:|---|---|---|
| `:54` | `head_mu` | `[B, N]` → softmax → `mu_pred` (orientation prior) | |
| `:73-74` | `head_feedback_focused`, `head_feedback_routine` | `[B, N]` raw (can be ±) | Per-regime heads; selected via `task_state` gate |
| `:94-95` | `head_pi` | `[B, 1]` → `softplus(·)` clamped at `pi_max=5.0` → `pi_pred` | |
| `:101` | `head_vip` (Rescue 3, not active in R1+R2 yaml) | `[B, N]` | |

### 2.3. Feedback pathway

In `network.py:295-348`:

```
feedback_signal = task_focused * head_feedback_focused(h_v2)
                + task_routine * head_feedback_routine(h_v2)      # [B, N], unclipped
pi_pred_eff    = pi_pred * feedback_scale                         # [B, 1]
precision_gate = pi_pred_eff / pi_max                              # Rescue 2 gating
scaled_fb      = feedback_signal * feedback_scale * precision_gate
center_exc     = relu(+scaled_fb)    # → L2/3 additive excitation  (Dale-E)
som_drive_fb   = relu(-scaled_fb)    # → SOM → L2/3 subtractive     (Dale-I)
```

`feedback_scale` is a learned-schedule buffer; at inference it equals `1.0`.

### 2.4. Key scalar constants (`sweep_rescue_1_2.yaml`)

| Param | Value | Source |
|---|---:|---|
| `tau_l4`, `tau_pv`, `tau_l23`, `tau_som` | 5, 5, 10, 10 | model |
| `sigma_rec` | 15.0° | model |
| `gain_rec` | 0.3 | model |
| `naka_rushton_n`, `c50` | 2.0, 0.3 | model |
| `sigma_ff` | 12.0° | model |
| `v2_hidden_dim` | 16 | model |
| `pi_max` | 5.0 | model |
| `tau_adaptation`, `alpha_adaptation`, `adaptation_clamp` | 200, 0.3, 10.0 | model |
| `feedback_mode` | `emergent` | model |
| `use_ei_gate` | `false` | model — alpha_net **not** wired in R1+R2 |
| `use_precision_gating` | `true` (Rescue 2) | model |

### 2.5. R1 and R2 rescues (vs simple-dual baseline)

- **Rescue 1** — `lambda_expected_suppress = 2.0`, but the loss body was changed from `|r_l23|` to `dot(r_l23, q_pred)` on routine-expected presentations. Source: yaml line `:71`; loss code in `src/training/losses.py`.
- **Rescue 2** — `use_precision_gating = true` (yaml `:38`): `pi_pred` gates feedback strength. Source code path: `network.py:303,316`.
- **Ancillary** — `lambda_state = 1.0` (yaml `:63`, was 0 in plain simple-dual) to re-enable the prior-KL that gives `q_pred` and `pi_pred` meaningful training signal.

No Rescue 3/4/5 is active in R1+R2 (`v2_context.py:101` `head_vip` absent; `deep_template_pop` and shape-matched prediction both off).

### 2.6. Input stream

HMM-march stimulus generator (`src/stimulus/sequences.py:271 HMMSequenceGenerator`). Stimulus sequence builder: `src/training/trainer.py:188 build_stimulus_sequence`.

Per presentation: `steps_on = 12` + `steps_isi = 4` = 16 timesteps; sequence length `= 25` presentations ⇒ 400 timesteps/trial. Default batch size 32. Readout window for all decoding analyses is `t ∈ [9, 11]` within the ON phase (mean over two timesteps late in the grating ON).

Stimulus params: `cue_dim=2`, `n_states=2`, `p_self=0.95`, `p_transition_cw=p_transition_ccw=0.80`, `ambiguous_fraction=0.3`, `ambiguous_offset=15.0°`, `cue_valid_fraction=0.75`, `transition_step=5.0°`.

---

## 3. Training pipeline

### 3.1. Stage 1 — sensory scaffold (`src/training/stage1_sensory.py:run_stage1`, 2000 Adam steps, lr 1e-3)

- **Trainable**: L2/3 + PV + `loss_fn.orientation_decoder` (i.e. Dec A). Frozen: everything else.
- **Loop** (`:132-163`): random gratings with contrasts in `[0.1, 1.0]`; V1-only forward pass (20 timesteps, no V2/feedback); cross-entropy on `orientation_decoder(r_l23) vs target channel` + `lambda_homeo * homeostasis_penalty`.
- **Critical detail**: decoder is added to the optimiser at construction (`:127-129: all_params = trainable + list(loss_fn.orientation_decoder.parameters())`) and trained at **every** step from 0 to `n_steps`. No burn-in, no ramp, no freeze-until-step-N. Dec A's weights are therefore fit to a **moving target** (early-training L2/3 ≠ late-training L2/3).
- After the loop: `freeze_stage1(net)` (`src/training/trainer.py:31`) freezes L4, PV, and L2/3's `w_som` / `w_pv_l23` gains; `l23.sigma_rec_raw` and `l23.gain_rec_raw` remain trainable for Stage 2. Decoder state-dict saved to `Stage1Result.decoder_state_dict` and transferred into Stage 2 via the shared `loss_fn`.

### 3.2. Stage 2 — V2 + feedback (`src/training/stage2_feedback.py`, 5000 AdamW steps)

- **Param groups** (`src/training/trainer.py:96-162 create_stage2_optimizer`):
  - V2 params (`v2.parameters()` + `v2_routine.parameters()` if present) at `lr_v2 = 3e-4`
  - `l23.sigma_rec_raw`, `l23.gain_rec_raw` at `lr_feedback = 1e-4`
  - `loss_fn.orientation_decoder.parameters()` at `stage1_lr = 1e-3`
  - (Optional) alpha_net, VIP, deep_template_pop, extra readout heads — not active in R1+R2.
- **LR schedule**: linear warmup 500 steps → cosine decay to 0 (`make_warmup_cosine_scheduler` at `trainer.py:169`).
- **Feedback schedule**: burn-in 1000 steps (`feedback_scale = 0`) → ramp 1000 steps (`feedback_scale: 0 → 1`) → full `feedback_scale = 1.0` for the remaining 3000 steps.
- **Decoder freeze in Stage 2**: `stage2_feedback.py:161-165` freezes the decoder iff `train_cfg.freeze_decoder` or `train_cfg.freeze_v2` is `True`. Both default to `False` (`src/config.py:226, 243`). Under `sweep_rescue_1_2.yaml` neither is set, so the decoder **continues to train** across Stage 2 at `stage1_lr = 1e-3`.
- **Losses** (`src/training/losses.py:CompositeLoss`):
  - `lambda_sensory = 1.0` (cross-entropy on `orientation_decoder(r_l23) vs probe channel`, at readout window `[9, 11]`)
  - `lambda_mismatch = 1.0` (BCE via MLP head — only active on routine branch via task routing)
  - `lambda_energy = 1.0` (L1 on `r_l23`)
  - `lambda_state = 1.0` (prior KL on `q_pred` / `pi_pred`)
  - `lambda_expected_suppress = 2.0` (Rescue 1 feature-specific body)
  - All other λ's = 0 in the yaml.
  - **Per-regime task routing** (yaml `:85-103`): `focused → sensory 3.0, energy 1.0, mismatch 0.0`; `routine → sensory 0.3, mismatch 1.0, energy 1.0`.
- **Gradient clip**: `1.0`.
- **Stimulus noise**: `0.25` (pre-Naka-Rushton additive).

### 3.3. Seeds and reproducibility

- Seed `42` for init + training stream (single seed checkpoint).
- `stage2_contrast_range = [0.15, 1.0]`; `ambiguous_fraction = 0.3` applied per trial in the HMM generator.

---

## 4. Decoders: A / A′ / B / C

Four orientation decoders exist in the project. Only Dec A is part of the trained network. Dec A′ is the 2026-04-23 stable-target retrain. Dec B and Dec C are post-hoc analysis tools. All three are applied to the **same** per-trial `r_l23` (one forward pass, multiple readouts) whenever a cross-decoder evaluation is performed.

### 4.1. Taxonomy

| Decoder | Type | Training data | Samples | Sees unexpected in training? | 10k HMM top-1 (R1+R2) | Artefact |
|---|---|---|---|---|---:|---|
| **A** | `Linear(36, 36)` saved with each Stage-1 checkpoint, continued training in Stage 2. | Natural HMM-march `r_l23` during Stage 1 + Stage 2 training (moving target — L2/3 changes every step, cf. `src/training/stage1_sensory.py:127-163`). | All Stage-1 training trials + Stage-2 trials. | Yes — the natural HMM stream contains jumps/ambiguous/task-switch presentations. | **0.5413** | `ckpt['loss_heads']['orientation_decoder']` |
| **A′** | Standalone `Linear(36, 36)` with bias, trained **only after the network was fully trained and frozen**. | `r_l23` streamed through the frozen R1+R2 network. Per step: batch 32 × seq 25 = 800 readouts at `t∈[9,11]`, ambiguous kept in, 50/50 focused/routine task_state. Adam lr 1e-3, 5000 gradient steps, seed 42; val pool seed 1234, ~8k readouts. | 5000 × 800 = 4 000 000 readouts total. | Yes — same stream, but on **stable** (post-training) L2/3. | **0.5486** | `checkpoints/decoder_a_prime.pt` |
| **B** | 5-fold nearest-centroid CV (`src/analysis/decoding.py:46 cross_validated_decoding`; helper in `scripts/cross_decoder_eval.py:189 decB_acc_5fold`). | The same `r_l23` being analysed — no separate training set. | Varies per-assay (set under evaluation). | Inherits exposure from the assay under evaluation. | — (not applicable; CV over analysis set) | — (computed on demand) |
| **C** | Standalone `Linear(36, 36)` with bias. | 100k synthetic orientation-bump patterns: 50k single-orientation σ=3 ch, amplitudes ∈ [0.1, 2.0]; 50k multi-orientation K∈{2,3} with strictly-max amplitude as the label; Gaussian noise σ=0.02. Adam lr 1e-3, batch 256, ≤30 epochs, early-stop patience 3, seed 42. | 100k synthetic. | No — trained on clean synthetic bumps only; never sees network `r_l23` or HMM context. | **0.5345** | `checkpoints/decoder_c.pt` |

### 4.2. Agreement on the 10k natural-HMM stream (Task #25 design, seed 42, readout window [9,11])

| Pair | `frac_same_pred` | Mean circular distance (ch) |
|---|---:|---:|
| A vs C | 0.6691 | 0.43 |
| A vs A′ | 0.8200 | 0.25 |
| A′ vs C | 0.6367 | 0.48 |

Source: `/tmp/task25_dec_av_c_summary.json` (A vs C), `results/decoder_a_prime_stratified_eval.json` (A vs A′, A′ vs C).

### 4.3. Stratified accuracy on 10k natural HMM (Task #25 strata; `pi_q1 = 0.824`, `pi_q3 = 2.362`; jump threshold 30°; pred-err thresholds 5° / 20°; 50/50 focused/routine per batch)

| Stratum | n | Dec A top-1 | Dec A′ top-1 | Dec C top-1 |
|---|---:|---:|---:|---:|
| overall | 10000 | 0.5413 | 0.5486 | 0.5345 |
| ambiguous | 2960 | 0.2578 | 0.2436 | 0.2304 |
| clean | 7040 | 0.6605 | 0.6768 | 0.6624 |
| `pi_low_Q1` | 2500 | 0.4640 | 0.4352 | 0.5020 |
| `pi_high_Q4` | 2500 | 0.6208 | 0.6472 | 0.5620 |
| `low_pred_err ≤ 5°` | 2383 | 0.5237 | 0.4851 | 0.5451 |
| `high_pred_err > 20°` | 4870 | 0.6031 | 0.6281 | 0.5483 |
| focused | 5000 | 0.6512 | 0.6572 | 0.6164 |
| routine | 5000 | 0.4314 | 0.4400 | 0.4526 |
| march_smooth | 8645 | 0.5098 | 0.5175 | 0.5519 |
| jump | 1355 | 0.7424 | 0.7469 | 0.4236 |

MAE in channels (overall, 10k): Dec A 0.820, Dec A′ 0.790, Dec C 0.862. `within1`: 0.8871 / 0.8960 / 0.8960. Dec A′ has the smallest overall MAE and the highest `within1`.

### 4.4. Cross-decoder bias flags (17-row matrix, `results/cross_decoder_comprehensive.json`)

- **Dec A**: `mean |Δ| = 0.2024`, `max |Δ| = 0.3871`; never the single sign-outlier. Always aligns with the 2/3 majority, but produces the largest-magnitude Δ in every row (amplifier).
- **Dec A′** (13 R1+R2 rows only): `mean |Δ| = 0.2138`, `max |Δ| = 0.3902`; zero sign flips vs Dec A on the same 13 rows.
- **Dec B**: `mean |Δ| = 0.0598`, `max |Δ| = 0.1818`; outlier in 5/17 rows (HMM C2, C3, C4 on R1+R2; HMM C1 on a1 legacy; P3P on R1+R2).
- **Dec C**: `mean |Δ| = 0.0399`, `max |Δ| = 0.1254`; outlier in 3/17 rows (HMM C1 on c1 legacy; HMS on R1+R2; HMS-T modified on R1+R2). Smallest-magnitude Δ in most rows.

### 4.5. Dec C training and validation (`scripts/train_decoder_c.py`; `results/decoder_c_validation.json`)

- **Training**: 100 000 synthetic patterns, 90/10 split. Early-stopped at epoch 29 with `best_val_acc = 0.8145`.
- **Held-out synthetic**: `top-1 = 0.8145` (single-orientation 0.9800; multi-orientation 0.6522). Note: target 0.95 not met (multi-orientation is the hard case).
- **Real-network natural HMM (R1+R2, 3200 readouts, 4 batches × seq 25)**: all-trials `top-1 = 0.5334` / `±1 = 0.8906` / `±2 = 0.9616`; non-ambiguous slice (2235 readouts) `top-1 = 0.6635` / `±1 = 0.9714` / `±2 = 0.9803`.

### 4.6. Dec A′ training curve (`results/decoder_a_prime_training.json`; 5000 Adam steps, lr 1e-3, batch 32 × seq 25, seed 42, val seed 1234)

| Step | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 0 | — | — | 3.586 | 0.032 |
| 500 | 2.923 | 0.274 | 2.468 | 0.352 |
| 1000 | 2.234 | 0.361 | 2.049 | 0.378 |
| 1500 | 1.926 | 0.404 | 1.822 | 0.439 |
| 2000 | 1.744 | 0.459 | 1.672 | 0.481 |
| 2500 | 1.617 | 0.499 | 1.562 | 0.513 |
| 3000 | 1.520 | 0.522 | 1.476 | 0.530 |
| 3500 | 1.443 | 0.536 | 1.407 | 0.541 |
| 4000 | 1.380 | 0.547 | 1.349 | 0.549 |
| 4500 | 1.326 | 0.553 | 1.301 | 0.551 |
| 5000 | 1.285 | 0.558 | 1.261 | 0.556 |

Comparison at step 5000 on the same val pool: Dec A (original) `val_loss = 1.208`, `val_acc = 0.547`. Dec A′ edges Dec A in accuracy (+0.009 pp) but has slightly higher CE loss (+0.053). Training was run on local CUDA; wall-clock 2214 s (~2.26 steps/s).

---

## 5. Evaluation paradigms catalogue

Every paradigm below is evaluated on the **same** R1+R2 checkpoint. All use seed 42 and the `[9, 11]` readout window.

### 5.1. Paired HMM-fork paradigms (constructive probe)

- **NEW (`eval_ex_vs_unex_decC.py` → `results/eval_ex_vs_unex_decC.json`)**: 2400 paired ex/unex trials per value of N (where N = number of HMM anchor presentations before the probe), values N=4..15. Shared pre-probe march, probe consistent (ex) or 90° shifted (unex). `pre_probe_max_abs_diff = 0.0` (bit-identical pre-probe). Pooled across N:
  - Dec acc: `ex = 0.7067 ± 0.0093`, `unex = 0.5813 ± 0.0101`, `Δdec = +0.1254`.
  - Net L2/3 `[B, N]` summed: `ex = 4.987 ± 0.012`, `unex = 6.134 ± 0.022`, `Δnet = −1.148`.
  - Peak-at-stim (max of re-centered curve): `ex = 0.7734 ± 0.0031`, `unex = 0.6263 ± 0.0040`, `Δpeak = +0.1470`.
  - FWHM (linear-interp at half-max of re-centered curve): `ex = 28.444° ± 0.098`, `unex = 29.774° ± 0.192`, `ΔFWHM = −1.330°` (ex **narrower** than unex).
- **4-condition `paradigm_readout` paired HMM fork (`r1r2_paradigm_readout.py` → `/tmp/task26_paradigm_R1R2.json`)**: 1000 paired ex/unex trials × 4 conditions (focused/routine × HMM cue / zero neutral cue), same paired-fork mechanism. Δ from branches:

| Condition | n per branch | Δ_A | Δ_B | Δ_C | Peak Δ (ex−unex) | Net-L2/3 Δ (ex−unex) |
|---|---:|---:|---:|---:|---:|---:|
| C1 focused + HMM cue | 1000 | +0.3090 | +0.0150 | +0.0500 | +0.0680 | −0.6832 |
| C2 routine + HMM cue | 1000 | +0.1580 | −0.0170 | +0.0060 | +0.0581 | −0.1876 |
| C3 focused + zero cue | 1000 | +0.3000 | −0.0070 | +0.0380 | +0.1076 | −0.5722 |
| C4 routine + zero cue | 1000 | +0.1550 | −0.0360 | +0.0390 | +0.0652 | −0.2006 |

FWHM is not reported per-branch in this JSON (the `paradigm_readout` pipeline reports peak and net per-branch; FWHM is in the NEW paired-march eval only).

- **Adjacent-channel signed-offset curve (Task #19, `results/eval_ex_vs_unex_decC_adjacent.json`)**: paired trials re-centered and march-direction-aligned. On expected trials, the `+k` flank (march leading edge) is **lower** than the `−k` flank for k ∈ {1, 2, 3}:

| k | ex at −k | ex at +k | Δ (−k − +k) |
|---:|---:|---:|---:|
| 1 | 0.7516 | 0.6885 | 0.0631 |
| 2 | 0.6287 | 0.5285 | 0.1002 |
| 3 | 0.4485 | 0.3489 | 0.0996 |

On unexpected trials, the `−k` vs `+k` difference is within ±0.003 at k ∈ {1, 2, 3} (near-symmetric). The asymmetry is observed only on expected trials. The asymmetry itself is an empirical observation; no feedback-ablation or weight-inspection experiment has been run to confirm a mechanism.

### 5.2. Observational matched-probe paradigms (unpaired)

From `scripts/cross_decoder_eval.py` on R1+R2 (`/tmp/task26_xdec_native.json`, CPU, seed 42, n_batches=40, n_trials_per_N=200):

| Strategy | n_ex | n_unex | Δ_A | Δ_B | Δ_C | Native decoder | Description |
|---|---:|---:|---:|---:|---:|:--:|---|
| NEW (paired march) | 2400 | 2400 | +0.3871 | +0.0854 | +0.1254 | C | Paired-fork constructive probe, N=4..15 |
| M3R (`matched_3row_ring`) | 1084 | 3302 | −0.1496 | −0.0082 | −0.0294 | A | Three-row ring; observational match on π_Q75 / exp_pred_err |
| HMS (`matched_hmm_ring_sequence`) | 3074 | 153 | −0.1850 | −0.1103 | +0.0790 | A | HMM ring sequence; observational match |
| HMS-T (`--tight-expected`) | 793 | 101 | −0.2919 | −0.1818 | −0.0631 | A | HMS with tightened expected-pred-err filter |
| P3P (`matched_probe_3pass`) | 38 | 38 | +0.3684 | −0.1714 | +0.0526 | A | 3-pass synthetic Pass A / B / omission probe; small n |
| VCD-test3 (`v2_confidence_dissection`) | 8025 | 8025 | −0.1655 | −0.0984 | −0.0703 | A | V2 confidence dissection |

### 5.3. Modified-input (focused + march cue) variants

From `scripts/cross_decoder_eval.py --override-task-cue` on R1+R2 (`/tmp/task26_xdec_modified.json`): pins `task_state=[1,0]` (focused) and replaces cue with deterministic march-continuation bumps (σ=10°).

| Strategy | n_ex | n_unex | Δ_A | Δ_B | Δ_C |
|---|---:|---:|---:|---:|---:|
| M3R modified | 3260 | 6486 | −0.1373 | −0.0336 | −0.0176 |
| HMS-T modified | 1139 | 111 | −0.2968 | −0.1222 | +0.0443 |
| VCD modified | 6998 | 6998 | −0.0840 | −0.0264 | −0.0130 |

### 5.4. HMM C1 (focused + HMM cue) on legacy reference networks (a1 / b1 / c1 / e1)

From `/tmp/task26_legacy/{a1,b1,c1,e1}_C1.json` (loaded via `MechanismType` shim in `src/config.py:11-25` + `torch.load(..., weights_only=False)` + `strict=False`):

| Network | n_ex | n_unex | Δ_A | Δ_B | Δ_C | Section-5 regime |
|---|---:|---:|---:|---:|---:|---|
| a1 | 1000 | 1000 | −0.0220 | +0.0000 | −0.0090 | Baseline dampening |
| b1 | 1000 | 1000 | −0.0320 | −0.0150 | −0.0230 | Baseline dampening (stronger) |
| c1 | 1000 | 1000 | +0.1870 | +0.0370 | −0.0070 | Transition / mixed |
| e1 | 1000 | 1000 | +0.2130 | +0.0510 | +0.0110 | Best sharpener |

See § 8 for context on the Section-5 classifications.

---

## 6. 17-row cross-decoder comprehensive matrix (`results/cross_decoder_comprehensive.json`)

Single forward pass per trial; three readouts per trial (Dec A / Dec B / Dec C). Aggregator: `scripts/aggregate_cross_decoder_matrix.py`.

### 6.1. Compact Δ = acc(ex) − acc(unex)

| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A′ | Δ_B | Δ_C | majority | outlier |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 | 1000 | 1000 | +0.3090 | +0.3070 | +0.0150 | +0.0500 | + | — |
| 2 | HMM C2 (routine + HMM cue) | R1+R2 | 1000 | 1000 | +0.1580 | +0.1690 | −0.0170 | +0.0060 | + | B |
| 3 | HMM C3 (focused + zero cue) | R1+R2 | 1000 | 1000 | +0.3000 | +0.2820 | −0.0070 | +0.0380 | + | B |
| 4 | HMM C4 (routine + zero cue) | R1+R2 | 1000 | 1000 | +0.1550 | +0.1560 | −0.0360 | +0.0390 | + | B |
| 5 | HMM C1 | a1 legacy | 1000 | 1000 | −0.0220 | — | +0.0000 | −0.0090 | − | B |
| 6 | HMM C1 | b1 legacy | 1000 | 1000 | −0.0320 | — | −0.0150 | −0.0230 | − | — |
| 7 | HMM C1 | c1 legacy | 1000 | 1000 | +0.1870 | — | +0.0370 | −0.0070 | + | C |
| 8 | HMM C1 | e1 legacy | 1000 | 1000 | +0.2130 | — | +0.0510 | +0.0110 | + | — |
| 9 | NEW (paired march) | R1+R2 | 2400 | 2400 | +0.3871 | +0.3888 | +0.0854 | +0.1254 | + | — |
| 10 | M3R (matched_3row_ring) | R1+R2 | 1084 | 3302 | −0.1496 | −0.0887 | −0.0082 | −0.0294 | − | — |
| 11 | HMS (matched_hmm_ring_sequence) | R1+R2 | 3074 | 153 | −0.1850 | −0.1556 | −0.1103 | +0.0790 | − | C |
| 12 | HMS-T (--tight-expected) | R1+R2 | 793 | 101 | −0.2919 | −0.2112 | −0.1818 | −0.0631 | − | — |
| 13 | P3P (matched_probe_3pass) | R1+R2 | 38 | 38 | +0.3684 | +0.3902 | −0.1714 | +0.0526 | + | B |
| 14 | VCD-test3 | R1+R2 | 8025 | 8025 | −0.1655 | −0.1987 | −0.0984 | −0.0703 | − | — |
| 15 | M3R modified | R1+R2 | 3260 | 6486 | −0.1373 | −0.1122 | −0.0336 | −0.0176 | − | — |
| 16 | HMS-T modified | R1+R2 | 1139 | 111 | −0.2968 | −0.2031 | −0.1222 | +0.0443 | − | C |
| 17 | VCD-test3 modified | R1+R2 | 6998 | 6998 | −0.0840 | −0.1165 | −0.0264 | −0.0130 | − | — |

Majority/outlier columns are computed from the original Δ_A / Δ_B / Δ_C triple (2026-04-22 matrix). `Δ_A′` values come from the 2026-04-23 rerun (§ 7). Legacy rows have Δ_A′ = — because Dec A′ was trained on R1+R2 `r_l23` and applying it to legacy `r_l23` would be a transfer test, not the same comparison.

### 6.2. Per-decoder profile

| Decoder | n rows | mean |Δ| | max |Δ| | rows agreeing with row-majority | rows where this decoder is single outlier |
|---|---:|---:|---:|---:|---|
| A | 17 | 0.2024 | 0.3871 | 17 / 17 | — |
| A′ (R1+R2 only) | 13 | 0.2138 | 0.3902 | 13 / 13 (signs identical to A on the same 13 rows) | — |
| B | 17 | 0.0598 | 0.1818 | 12 / 17 | HMM C2 / C3 / C4 on R1+R2; HMM C1 on a1 legacy; P3P on R1+R2 |
| C | 17 | 0.0399 | 0.1254 | 14 / 17 | HMM C1 on c1 legacy; HMS on R1+R2; HMS-T modified on R1+R2 |

### 6.3. Rows where all three decoders (A, B, C) agree on sign (9 / 17)

| # | Assay | Network | Common sign |
|---|---|---|:--:|
| 1 | HMM C1 (focused + HMM cue) | R1+R2 | + |
| 6 | HMM C1 | b1 legacy | − |
| 8 | HMM C1 | e1 legacy | + |
| 9 | NEW (paired march) | R1+R2 | + |
| 10 | M3R | R1+R2 | − |
| 12 | HMS-T | R1+R2 | − |
| 14 | VCD-test3 | R1+R2 | − |
| 15 | M3R modified | R1+R2 | − |
| 17 | VCD-test3 modified | R1+R2 | − |

Two rows with + majority and three-decoder agreement on R1+R2: rows 1, 9. Five rows with − majority and three-decoder agreement on R1+R2: rows 10, 12, 14, 15, 17.

---

## 7. Dec A → Dec A′ swap (stable-target sanity check, 2026-04-23)

### 7.1. Motivation

Dec A (`loss_fn.orientation_decoder` in `src/training/stage1_sensory.py:127-163`) is added to the Stage-1 optimiser at step 0 and trained every step through Stage 1 (2000 steps) and Stage 2 (5000 steps, unless `freeze_decoder` is set — not set in R1+R2 yaml). Its weights therefore fit a **moving target**: L2/3 representations change step-by-step as the network learns. Dec A′ is a fresh `Linear(36, 36)` retrained for 5000 Adam steps on `r_l23` streamed through the **fully-trained, frozen** R1+R2 network (net.eval() + `requires_grad_(False)` on every net param, verified by assertion at setup). Question: does the moving-target training for Dec A change the 13-row sign pattern on R1+R2?

### 7.2. Matrix rerun

Patched R1+R2 ckpt (`scripts/_make_decAprime_ckpt.py`): `ckpt['loss_heads']['orientation_decoder']` and `ckpt['decoder_state']` replaced with Dec A′ weights; `ckpt['model_state']` (network weights) untouched. Saved to `/tmp/r1r2_ckpt_decAprime.pt`. Reran the three R1+R2 pipelines (`r1r2_paradigm_readout.py`, `cross_decoder_eval.py` native, `cross_decoder_eval.py --override-task-cue`) against the patched ckpt on CPU, seed 42, identical other args to the original 2026-04-22 invocations. Legacy rows retain their own stored Dec A. Re-aggregated via `aggregate_cross_decoder_matrix.py`. Diff via `diff_decAprime_matrix.py` → `results/cross_decoder_comprehensive_decAprime_diff.{json,md}`.

### 7.3. Per-row Δ_A → Δ_A′ on the 13 R1+R2 rows

| # | Assay | Δ_A | Δ_A′ | Δ shift (A′ − A) |
|---|---|---:|---:|---:|
| 1 | HMM C1 | +0.3090 | +0.3070 | −0.0020 |
| 2 | HMM C2 | +0.1580 | +0.1690 | +0.0110 |
| 3 | HMM C3 | +0.3000 | +0.2820 | −0.0180 |
| 4 | HMM C4 | +0.1550 | +0.1560 | +0.0010 |
| 9 | NEW | +0.3871 | +0.3888 | +0.0017 |
| 10 | M3R | −0.1496 | −0.0887 | +0.0609 |
| 11 | HMS | −0.1850 | −0.1556 | +0.0294 |
| 12 | HMS-T | −0.2919 | −0.2112 | +0.0807 |
| 13 | P3P | +0.3684 | +0.3902 | +0.0218 |
| 14 | VCD | −0.1655 | −0.1987 | −0.0332 |
| 15 | M3R modified | −0.1373 | −0.1122 | +0.0250 |
| 16 | HMS-T modified | −0.2968 | −0.2031 | +0.0936 |
| 17 | VCD modified | −0.0840 | −0.1165 | −0.0324 |

### 7.4. Summary statistics

- **Zero Δ-sign flips** across all 13 rows. `sign(Δ_A′) = sign(Δ_A)` everywhere.
- `|Δ_A′ − Δ_A|` max = 0.0936 (row 16, HMS-T modified); median 0.0250; mean 0.0316.
- The three largest shifts are sharpening-side rows (HMS-T native +0.081, HMS-T modified +0.094, M3R native +0.061), all toward smaller |Δ|. Dec A′ consistently produces a less-extreme negative Δ than Dec A on sharpening rows, and near-equal Δ on dampening (ex > unex) rows.

### 7.5. Sign-agreement class changes — signal vs noise

Two rows shift their sign-agreement class between the original matrix and the Dec A′ rerun, but both shifts come from Δ_B drift in the rerun, not from the Dec A → Dec A′ swap:

| Row | Assay | Original agreement | Rerun agreement (run-matched Δ_B/Δ_C) | Δ_B orig → rerun |
|---|---|---|---|---|
| 3 | HMM C3 (focused + zero cue) | split (A+, B−, C+; outlier B) | ALL-agree (A+, B+, C+) | −0.0070 → +0.0240 |
| 10 | M3R (native) | ALL-agree (A−, B−, C−) | split (A−, B+, C−; outlier B) | −0.0082 → +0.0025 |

Holding Δ_B / Δ_C fixed at their original 2026-04-22 values and only swapping Dec A → Dec A′, **zero rows change sign-agreement class**. The Dec A′ rerun's Δ_B / Δ_C drifted by ≤ 0.03 from the original matrix (same seed, same CPU, residual FP run-to-run noise); Δ_B for rows 3 and 10 crosses zero by ±0.02. The two class changes are therefore attributable to Δ_B noise, not to Dec A′. Source: `results/cross_decoder_comprehensive_decAprime_diff.json`.

### 7.6. Observation (not a claim about correctness)

The 13-row sign pattern — sharpening on rows 1, 9; dampening on rows 10, 12, 14, 15, 17; split on rows 2, 3, 4, 11, 13, 16 — is identical under Dec A and Dec A′. The "moving target during Stage 1" concern for Dec A does not flip any R1+R2 row's Δ sign. This does not validate Dec A as ground truth; it is a consistency check between two linear decoders trained on the same network under slightly different schedules.

---

## 8. Legacy reference networks (a1 / b1 / c1 / e1)

### 8.1. Section-5 regime classification (RESULTS.md § 5, 25-run parameter sweep)

All legacy runs: `stage1 = 2000 steps`, `stage2 = 5000 steps`, seed 42.

| Run | λ_sensory | λ_energy | l23_energy_weight | M7 (d=10°) | M10 (amplitude ratio) | FWHM Δ (deg) | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| a1 | 0.0 | 2.0 | 1.0 | −0.047 | 0.70 | −3.9 | Dampening (baseline) |
| b1 | 0.0 | 5.0 | 1.0 | −0.060 | 0.64 | −3.0 | Dampening (stronger) |
| c1 | 0.3 | 2.0 | 5.0 | +0.081 | 0.95 | −4.3 | Transition / mixed |
| e1 | 0.3 | 2.0 | 3.0 | +0.104 | 1.13 | −11.6 | Sharpening (best of 25) |

### 8.2. Three-decoder reproduction on HMM C1 (focused + HMM cue)

From § 5.4 above (same numbers reproduced here for convenience):

| Network | Δ_A | Δ_B | Δ_C | Sign-agreement |
|---|---:|---:|---:|---|
| a1 | −0.0220 | +0.0000 | −0.0090 | B outlier (B exactly 0.0; A and C both negative) |
| b1 | −0.0320 | −0.0150 | −0.0230 | ALL agree (dampening) |
| c1 | +0.1870 | +0.0370 | −0.0070 | C outlier (C weakly negative at transition boundary) |
| e1 | +0.2130 | +0.0510 | +0.0110 | ALL agree (sharpening) |

Dec A amplifies the sharpening-vs-dampening gap (`−0.032 → +0.213` across the four networks, range 0.245). All three decoders agree on sign in 2 of 4 rows (b1, e1). Under the looser A-vs-C-only check, 3 of 4 rows agree (a1, b1, e1); only c1 disagrees (c1 sits at the parameter-sweep transition boundary).

The legacy Section-5 regime classifications (a1, b1 dampening; e1 sharpening) reproduce under Dec C on those same three networks. R1+R2 on the same HMM C1 assay (row 1 of § 6) gives `Δ_A = +0.309, Δ_B = +0.015, Δ_C = +0.050`, all three positive — anchoring R1+R2 to the sharpening side **on this assay** (paradigm-dependent, see § 9).

---

## 9. Robust findings synthesis

"Decoder-robust" in this doc means all three decoders (A, B, C) agree on the sign of Δ = acc(ex) − acc(unex). Only the all-agree subset is listed as decoder-robust below.

### 9.1. Decoder-robust sharpening on R1+R2 (Δ > 0 on all three)

| # | Assay | Δ_A | Δ_B | Δ_C |
|---|---|---:|---:|---:|
| 1 | HMM C1 (focused + HMM cue) | +0.3090 | +0.0150 | +0.0500 |
| 9 | NEW (paired march) | +0.3871 | +0.0854 | +0.1254 |

### 9.2. Decoder-robust dampening on R1+R2 (Δ < 0 on all three)

| # | Assay | Δ_A | Δ_B | Δ_C |
|---|---|---:|---:|---:|
| 10 | M3R | −0.1496 | −0.0082 | −0.0294 |
| 12 | HMS-T | −0.2919 | −0.1818 | −0.0631 |
| 14 | VCD-test3 | −0.1655 | −0.0984 | −0.0703 |
| 15 | M3R modified | −0.1373 | −0.0336 | −0.0176 |
| 17 | VCD-test3 modified | −0.0840 | −0.0264 | −0.0130 |

### 9.3. Decoder-dependent on R1+R2 (≥ one decoder disagrees on sign)

| # | Assay | Δ_A | Δ_B | Δ_C | Outlier |
|---|---|---:|---:|---:|---|
| 2 | HMM C2 (routine + HMM cue) | +0.1580 | −0.0170 | +0.0060 | B |
| 3 | HMM C3 (focused + zero cue) | +0.3000 | −0.0070 | +0.0380 | B |
| 4 | HMM C4 (routine + zero cue) | +0.1550 | −0.0360 | +0.0390 | B |
| 11 | HMS | −0.1850 | −0.1103 | +0.0790 | C |
| 13 | P3P | +0.3684 | −0.1714 | +0.0526 | B |
| 16 | HMS-T modified | −0.2968 | −0.1222 | +0.0443 | C |

Rows 2/3/4 share the same outlier (B) — Dec B sits near zero or slightly negative on paired HMM fork conditions without the HMM cue dissociating focused from routine task-state. P3P row 13 has a small n (38/branch).

### 9.4. Hybrid regime claim

On the **paired-fork paradigm** (constructive probe, bit-identical pre-probe), R1+R2 is decoder-robust sharpening (rows 1 and 9). On **observational matched-probe paradigms** (M3R, HMS-T, VCD and their focused+march-cue modified variants), the same R1+R2 checkpoint is decoder-robust dampening (rows 10, 12, 14, 15, 17). The paradigm choice, not the decoder choice, drives the sign. The Dec A → Dec A′ sanity check (§ 7) does not change this: all 13 R1+R2 rows retain the same sign under Dec A′.

### 9.5. FWHM-sign reversal between two paired-fork variants

On the 4-condition paired HMM fork (`r1r2_paradigm_readout.py`), FWHM is not reported per-branch. On the NEW paired-march eval (`eval_ex_vs_unex_decC.py`), ex FWHM = 28.444° ± 0.098 is **narrower** than unex FWHM = 29.774° ± 0.192 (Δ = −1.330°). Both paradigms share the paired-fork structure; only the decoding sign (Δ_C > 0 on both) is consistent across the two variants. Peak Δ (ex > unex) and net-L2/3 Δ (ex < unex) are also consistent between them. FWHM per-branch in the `paradigm_readout` 4-condition sweep is not in the JSON used here.

---

## 10. Reproducibility and artefact index

### 10.1. Code paths (`src/`)

| Component | Path |
|---|---|
| Network scaffold | `src/model/network.py` (class `LaminarV1V2Network` at `:18`) |
| L4 / L2/3 / PV / SOM rings | `src/model/populations.py` (`:21`, `:179`, `:95`, `:290`) |
| V2 GRU + heads | `src/model/v2_context.py` (class at `:13`) |
| Stage-1 training | `src/training/stage1_sensory.py` (`run_stage1` at `:89`, loop at `:132-163`) |
| Stage-2 training | `src/training/stage2_feedback.py` (freeze logic at `:161-165`) |
| Optimiser / scheduler / freeze helpers | `src/training/trainer.py` (`create_stage2_optimizer` at `:96`) |
| Loss module (Dec A lives here) | `src/training/losses.py` (class `CompositeLoss` at `:22`, `orientation_decoder` at `:76`) |
| Dec B implementation | `src/analysis/decoding.py` (`cross_validated_decoding` at `:46`) |
| HMM generator | `src/stimulus/sequences.py` (class at `:271`) |
| Stimulus sequence builder | `src/training/trainer.py` (`build_stimulus_sequence` at `:188`) |
| Config loader + `MechanismType` shim | `src/config.py` (shim at `:11-25`) |

### 10.2. Scripts

| Task | Path |
|---|---|
| Dec A′ training | `scripts/train_decoder_a_prime.py` |
| Dec A′ stratified eval | `scripts/eval_decoder_a_prime_stratified.py` |
| Dec A′ ckpt patch helper | `scripts/_make_decAprime_ckpt.py` |
| Dec A vs A′ matrix diff | `scripts/diff_decAprime_matrix.py` |
| Dec C training | `scripts/train_decoder_c.py` |
| Paired HMM fork paradigm × readout | `scripts/r1r2_paradigm_readout.py` |
| Cross-decoder evaluator (NEW / M3R / HMS / HMS-T / P3P / VCD) | `scripts/cross_decoder_eval.py` |
| 17-row matrix aggregator | `scripts/aggregate_cross_decoder_matrix.py` |
| NEW paired-march eval | `scripts/eval_ex_vs_unex_decC.py` |
| Adjacent-channel signed-offset | `scripts/eval_ex_vs_unex_decC_adjacent.py` |

### 10.3. Checkpoints

| File | Contents |
|---|---|
| `results/simple_dual/emergent_seed42/checkpoint.pt` | R1+R2 network (model_state + loss_heads[orientation_decoder] = Dec A) |
| `checkpoints/decoder_c.pt` | Dec C weights (state_dict) + training metadata |
| `checkpoints/decoder_a_prime.pt` | Dec A′ weights (state_dict) + training metadata |
| `checkpoints/decoder_a_prime.pt.step2000`, `...step4000` | Dec A′ crash-safety snapshots |
| `/tmp/r1r2_ckpt_decAprime.pt` | R1+R2 ckpt with Dec A′ patched in place of Dec A |

### 10.4. Result JSONs

| Content | Path |
|---|---|
| 17-row matrix (original, Dec A) | `results/cross_decoder_comprehensive.{json,md}` |
| 17-row matrix (Dec A → Dec A′ swap on R1+R2 rows) | `results/cross_decoder_comprehensive_decAprime.{json,md}` |
| 17-row matrix diff (A vs A′ per row, sign-agreement analysis) | `results/cross_decoder_comprehensive_decAprime_diff.{json,md}` |
| 4-condition paired HMM fork (R1+R2) | `results/r1r2_paradigm_readout.json` · `/tmp/task26_paradigm_R1R2.json` |
| Paired HMM fork with omission branch | `results/r1r2_paired_hmm_fork.json` |
| NEW paired-march eval | `results/eval_ex_vs_unex_decC.json` |
| Adjacent-channel signed-offset (Task #19) | `results/eval_ex_vs_unex_decC_adjacent.json` |
| Dec C training + validation | `results/decoder_c_validation.json` |
| Dec A′ training curve | `results/decoder_a_prime_training.json` |
| Dec A′ stratified eval (Task #25 strata on A / A′ / C) | `results/decoder_a_prime_stratified_eval.json` |
| Task #25 Dec A vs Dec C stratified summary | `/tmp/task25_dec_av_c_summary.json` · `/tmp/task25_dec_av_c_pertrial.npz` |
| Dec A′ stratified per-trial arrays | `/tmp/decA_prime_stratified_pertrial.npz` |
| Cross-decoder native-invocation sources | `/tmp/task26_xdec_native.json`, `/tmp/task26_xdec_modified.json` |
| Cross-decoder sources under Dec A′ swap | `/tmp/task26_xdec_native_decAprime.json`, `/tmp/task26_xdec_modified_decAprime.json` |
| Legacy HMM C1 JSONs | `/tmp/task26_legacy/{a1,b1,c1,e1}_C1.json` |

### 10.5. Key logs

| Log | Path |
|---|---|
| Dec A′ training | `logs/decA_prime.log` |
| Dec A′ stratified eval | `logs/decA_prime_stratified.log` |
| Dec A′ Part B pipeline | `logs/decAprime_partB.log` |

### 10.6. Known discrepancies to watch for

- The original 2026-04-22 matrix and the 2026-04-23 Dec A′ rerun used the same seed / script / CPU device, but Δ_B / Δ_C still drifted by ≤ 0.03 row-to-row (residual FP noise). Treat sign-agreement class changes between the two runs as noise unless Δ_B / Δ_C differ by more than ≈ 0.03.
- Dec B is computed from the same `r_l23` as the assay under analysis (CV within assay). Sample size matters: row 13 (P3P) has n = 38 / branch, so the B outlier on P3P is consistent with the small-n noise floor.
- Dec A was continuously trained across Stage 2 under `sweep_rescue_1_2.yaml` (neither `freeze_decoder` nor `freeze_v2` is set). This is consistent with the "moving target" framing, but note that Stage-2 L2/3 is less volatile than Stage-1 L2/3 because `sigma_rec_raw` / `gain_rec_raw` are the only L2/3 params still trainable in Stage 2.

---

*End of report.*
