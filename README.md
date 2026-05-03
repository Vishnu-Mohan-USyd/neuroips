# Laminar V1–V2 Expectation Suppression Model

A minimal laminar V1–V2 rate model in PyTorch, trained with BPTT, used to study
how top-down expectation modulates early visual cortex activity. The repo
contains the model, training pipeline, sweep configs, analysis scripts, and the
writeups for several lines of investigation.

## Key branches

| Branch | What's on it |
|---|---|
| `main` | Stable baseline. |
| `single-network-dual-regime` | Active development of the single-network dual-regime model and the simple_dual checkpoints. |
| `failed-dual-regime-experiments` | Architectural rescue chain (R1+R2, R3, R4, R5) attempted to recover task-state-selective sharpening AND dampening from one architecture. See `docs/rescues_1_to_4_summary.md`. |
| `dampening-analysis` | Re-centered tuning-curve analysis + figures + corrective writeup on the rescue results. **Canonical default checkpoint on this branch is R1+R2** (`simple_dual/emergent_seed42/checkpoint.pt` on the remote) for all expectation-suppression / dampening-vs-sharpening analyses (set 2026-04-17). Network_mm / Network_both remain valid for the per-regime feedback question but are no longer the default model for ex-vs-unex analysis. |

## Key docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — model architecture, loss table, optional rescue-chain config flags, full three-decoder taxonomy (Dec A / B / C with training data, sample counts, sees-unex status, stratified strengths) and cross-decoder bias flags.
- [`RESULTS.md`](RESULTS.md) — original 25-run parameter sweep (§ 5, three emergent regimes), rescue-chain cross-checkpoint summary (§ 9), R1+R2 Decoder C paired ex/unex eval (§ 10), **cross-decoder comprehensive matrix (§ 11)**, **paired HMM fork paradigm × readout (§ 12)**, **legacy reference networks (§ 13)**, and the **robust findings summary (§ 14, 2026-04-22)**.
- [`docs/project_summary.md`](docs/project_summary.md) — comprehensive project history, with §15 (cross-decoder matrix), §16 (paradigm × readout), §17 (legacy references), §18 (robust findings — R1+R2-is-hybrid framing) as the current load-bearing sections.
- [`docs/rescues_1_to_4_summary.md`](docs/rescues_1_to_4_summary.md) — full per-rescue writeup, the 2026-04-13 re-centered analysis update, and the 2026-04-17 Decoder A artefact note.
- [`docs/research_log.md`](docs/research_log.md) — chronological research log; latest entry (2026-04-22) is the cross-decoder matrix + legacy ref reproduction; previous (2026-04-19) is the paired HMM fork paradigm × readout analysis.
- [`docs/training_experiments_analysis_design.md`](docs/training_experiments_analysis_design.md) — baseline training pipeline / paradigm / analysis spec.

## Headline finding (rescue chain)

R4 (DeepTemplate + error-mismatch) exhibits the cleanest Richter (2018) preserved-shape dampening signature: peak −15%, total −19%, FWHM matched within 1.5° between expected and unexpected trials. Main visual: `docs/figures/tuning_ring_recentered_r4.png`. The dampening is **task-state-invariant**, so the preregistered BOTH-regime criterion (focused → Kok sharpening, routine → Richter dampening from one network) is **not** met.

## Headline finding (R1+R2 expectation suppression is paradigm-dependent, 2026-04-22; decoder asymmetries added 2026-04-23/24)

Across the 17-row cross-decoder matrix covering paired-fork and observational paradigms (Tasks #15–#26), R1+R2 is a **hybrid network, not a single-regime network**:

- **Paired HMM fork paradigm (constructive probe, bit-identical pre-probe state):** **decoder-robust sharpening.** NEW eval on R1+R2: Δdec_A=+0.387, Δdec_B=+0.085, Δdec_C=+0.125; Δpeak ≈ +0.15, Δnet ≈ −1.15, ΔFWHM ≈ −1.3° (expected narrower, higher-peak, better-decoded, lower-activity). HMM C1–C4 (all four task-state × cue conditions) give Δdec_C ∈ {+0.088, +0.013, +0.045, +0.041} — all positive.
- **Matched-probe observational paradigms** — specifically M3R (`matched_3row_ring`), HMS-T (`matched_hmm_ring_sequence --tight-expected`), VCD (`v2_confidence_dissection`), plus the M3R and VCD `focused + march cue` variants: **decoder-robust dampening.** All three decoders give negative Δdec on these 5 rows; M3R Δdec_C = −0.029, HMS-T Δdec_C = −0.063, VCD Δdec_C = −0.070. The plain HMS variant (`matched_hmm_ring_sequence` without `--tight-expected`, row 11) and HMS-T modified (row 16) are NOT decoder-robust — Dec C flips positive on both — so they are excluded from this list.

The paradigm choice, not the decoder choice, drives the sign. The 2026-04-24 rerun extends the 17-row matrix to 7 decoder columns (A, A′, B, C, D-raw, D-shape, E); Tasks #5–#8 (2026-04-25 → 2026-05-03) extend it again to 8 decoders by adding 20k Dec A′ across all 5 nets and 20k Dec D (paired-fork balanced ex+unex training) on the 4 legacy nets. Closure: Section-5's original legacy classification (a1 / b1 dampening, c1 transitional, e1 best sharpener) is reconfirmed. The 5k-retrain sign flags on a1 / b1 HMM C1 (rows 5–6) are Adam @ 5k under-training; the 20k Dec A′ flags on the same rows are natural-HMM prior-bias overfitting at large ||W||; in both cases Dec D 20k (balanced ex+unex training, no prior bias to exploit) recovers Dec A's small-dampening direction. Details: `docs/research_log.md` 2026-05-03 entry, `docs/R1R2_full_report.md` § 9.6, `RESULTS.md` § 11 + § 14, `docs/project_summary.md` § 15 + § 18, `results/cross_decoder_comprehensive_20k_final.{json,md}`.

### Decoder summary (2026-04-24: 6 decoders in the 7-column matrix)

| Decoder | Training data | 10k natural-HMM top-1 (R1+R2) | Role |
|---|---|---:|---|
| **A** | Stage-1 `r_l23` during L2/3 training (moving target — L2/3 changes each step); continues training in Stage 2. | 0.5413 | Frozen network readout; amplifier (largest \|Δ\| but never the sign outlier across 17 rows). |
| **A′ (5k)** | 5000 Adam steps on `r_l23` from the **frozen, fully-trained** network with **50/50** focused/routine task_state (stable-target; 2026-04-23). Per-net for all 5 nets (R1+R2 + a1/b1/c1/e1). | 0.5486 (R1+R2) | Stable-target sanity check on Dec A. Zero Δ-sign flips across all 13 R1+R2 rows. **Deprecated for cross-decoder analysis** — under-trained on dampening nets a1 / b1 (top-1 0.36); use 20k variant for those. |
| **A′ (20k)** | 20 000 Adam steps; same data + protocol as 5k. Per-net (Tasks #6–#7, 2026-04-25). | 0.5729 / 0.6709 / 0.6625 / 0.5078 / 0.5319 (r1r2/a1/b1/c1/e1) | Optimisation-converged Dec A′. Exceeds Dec A on top-1 every net by +3 to +8 pp. **But** on a1 / b1 reads Δ_ex_unex = +0.21 / +0.18 — natural-HMM prior-bias overfitting at large ||W||, NOT genuine sharpening (Task #8 disambiguation). Use Dec D 20k for clean Δ on those nets. |
| **B** | 5-fold nearest-centroid CV on the analysis set. | — | Robustness control; outlier in 2 of 17 rows in 2026-04-24 rerun. |
| **C** | 100k synthetic orientation bumps (never sees `r_l23` during training). | 0.5345 | Preferred conservative readout for ex-vs-unex; smallest \|Δ\|, outlier in 4 of 17 rows. |
| **D-raw** | `Linear(36, 36)+bias` per-ckpt on FB-ON paired-fork neutral localizer (900/cell balanced, 72 k samples). Raw `r_l23` input. | 0.3634 | Amplitude-sensitive neutral readout. Per-net val balanced-acc: r1r2 0.187, a1 0.947, b1 0.922, c1 0.618, e1 0.556. |
| **D-shape** | Same protocol as D-raw but on `r_l23 / (r_l23.sum(1) + 1e-8)` (shape only). | 0.3726 | Shape-only readout. Signals amplitude-normalised pattern. Row 12 (HMS-T) gives Δ_D-shape = +0.166 while Δ_A = −0.303 — a Kok-style co-occurrence. |
| **D-raw / D-shape (20k)** | 20 000 Adam steps lr=1e-3 (no early stop, no wd). Re-uses cached paired-fork readouts from the 5k training. Per-net for a1 / b1 / c1 / e1 (Task #8, 2026-04-27). | (paired-fork in-distribution) | **The clean Δ_ex_unex test on dampening legacy nets.** Balanced ex+unex training removes natural-HMM prior asymmetry. On a1 / b1 HMM C1, Δ_D-shape(20k) = −0.052 / −0.044 — agreeing with Dec A's small-dampening direction and refuting the 20k Dec A′ positive flag. On c1 / e1: +0.084 / +0.067 — confirming Dec A's small sharpening. |
| **E** | Dec-A-spec (`Linear(36, 36)+bias`, Adam lr=1e-3, 5000 steps, seed 42) retrained per-ckpt on the **natural HMM stream with HMM's own stochastic task_state** (Markov p_switch=0.2 or Bernoulli-per-batch per each yaml); post-Stage-2. | 0.5467 | Close to Dec A on R1+R2 (+0.5 pp; `frac_same_pred(A′, E) = 0.9722`). On a1 / b1 caps at top-1 ~0.35 — Adam-@-5k under-training on dampened L2/3 (20k Dec A′ reaches 0.671 / 0.663). The 2 sign flags on a1 / b1 HMM C1 are 5k-undertrained-Adam Δ; the 20k Dec A′ flag at +0.21 / +0.18 on the same rows is natural-HMM prior-bias overfitting (Task #8 Dec D 20k balanced-training control reads −0.024 / −0.046, agreeing with Dec A). |

**Chance baseline = 1/36 ≈ 2.8%** for 36-way orientation classification. On R1+R2 the trained decoders A, A′ (5k), A′ (20k), E, C all sit at 0.53–0.57 (≈ 19–21× chance); D-raw / D-shape (5k or 20k) at 0.36–0.50 are lower because the paired-fork focused-only training distribution is OOD for the 10k HMM eval (50/50 focused/routine + ambiguous). Full taxonomy and cross-decoder bias flags: `ARCHITECTURE.md` § "Decoders". 17-row matrix with 8 decoder columns (A / A′_5k / A′_20k / B / C / D-raw / D-shape / E): `results/cross_decoder_comprehensive_20k_final.{json,md}`. Full account of the 5k → 20k → balanced-training Dec D analysis trajectory: `docs/research_log.md` 2026-05-03 entry, `docs/R1R2_full_report.md` § 9.6.

## Reproducing the rescue figures

Checkpoints live on the remote GPU (`vishnu@reuben-ml`):

| Checkpoint | Path on remote |
|---|---|
| Baseline (simple_dual) | `/home/vishnu/neuroips/simple_dual/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` |
| Rescue 1+2 | `/home/vishnu/neuroips/rescue_1_2/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` |
| Rescue 3 | `/home/vishnu/neuroips/rescue_3/freshstart/results/simple_dual/emergent_seed42/checkpoint.pt` |
| Rescue 4 | `/home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt` |

Example: regenerate the R4 re-centered ring figure (from the R4 tree on
the remote, which has the rescue config):

```bash
python3 scripts/plot_tuning_ring_extended.py \
    --config     config/sweep/sweep_rescue_4.yaml \
    --checkpoint /home/vishnu/neuroips/rescue_4/freshstart/results/rescue_4/emergent_seed42/checkpoint.pt \
    --fig1-name  docs/figures/tuning_ring_recentered_r4.png \
    --fig1-title "Re-centered ring — Rescue 4 (DeepTemplate + error-mismatch)" \
    --skip-fig2 \
    --n-batches 20
```

For the other checkpoints, swap `--config`, `--checkpoint`, `--fig1-name`,
`--fig1-title`. The probe-60° heatmap (`scripts/plot_tuning_ring_heatmap.py`)
and pooled-all-probes figure use the same data pipeline; see each script's
docstring for the exact CLI.

## Running the tests

```bash
python3 -m pytest tests/ -v
```

245 tests covering V1 circuit, V2 prior, feedback pathway, training pipeline,
config parsing, paradigms, analysis suite, and regression tests.
