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

The paradigm choice, not the decoder choice, drives the sign. The 2026-04-24 rerun extends the 17-row matrix to 7 decoder columns (A, A′, B, C, D-raw, D-shape, E) — the hybrid finding holds on all 13 R1+R2 rows. Two legacy rows (a1 and b1 HMM C1) show a **Dec A vs Dec E sign flip** (Δ_A ≈ −0.03 → Δ_E ≈ +0.03), concentrating the only Dec-A-vs-Dec-E disagreements of the full 17-row matrix on dampening legacy configs. Details: `RESULTS.md` § 11–§ 14, `docs/project_summary.md` § 15–§ 18, `ARCHITECTURE.md` § "Decoders", `results/cross_decoder_comprehensive_with_all_decoders.{json,md}`.

### Decoder summary (2026-04-24: 6 decoders in the 7-column matrix)

| Decoder | Training data | 10k natural-HMM top-1 (R1+R2) | Role |
|---|---|---:|---|
| **A** | Stage-1 `r_l23` during L2/3 training (moving target — L2/3 changes each step); continues training in Stage 2. | 0.5413 | Frozen network readout; amplifier (largest \|Δ\| but never the sign outlier across 17 rows). |
| **A′** | 5000 Adam steps on `r_l23` from the **frozen, fully-trained** R1+R2 network with **50/50** focused/routine task_state (stable-target; 2026-04-23). | 0.5486 | Stable-target sanity check on Dec A. Zero Δ-sign flips across all 13 R1+R2 rows; `frac_same_pred(A, A′) = 0.820`. R1+R2 only. |
| **B** | 5-fold nearest-centroid CV on the analysis set. | — | Robustness control; outlier in 2 of 17 rows in 2026-04-24 rerun. |
| **C** | 100k synthetic orientation bumps (never sees `r_l23` during training). | 0.5345 | Preferred conservative readout for ex-vs-unex; smallest \|Δ\|, outlier in 4 of 17 rows. |
| **D-raw** | `Linear(36, 36)+bias` per-ckpt on FB-ON paired-fork neutral localizer (900/cell balanced, 72 k samples). Raw `r_l23` input. | 0.3634 | Amplitude-sensitive neutral readout. Per-net val balanced-acc: r1r2 0.187, a1 0.947, b1 0.922, c1 0.618, e1 0.556. |
| **D-shape** | Same protocol as D-raw but on `r_l23 / (r_l23.sum(1) + 1e-8)` (shape only). | 0.3726 | Shape-only readout. Signals amplitude-normalised pattern. Row 12 (HMS-T) gives Δ_D-shape = +0.166 while Δ_A = −0.303 — a Kok-style co-occurrence. |
| **E** | Dec-A-spec (`Linear(36, 36)+bias`, Adam lr=1e-3, 5000 steps, seed 42) retrained per-ckpt on the **natural HMM stream with HMM's own stochastic task_state** (Markov p_switch=0.2 or Bernoulli-per-batch per each yaml); post-Stage-2. | 0.5467 | Close to Dec A on R1+R2 (+0.5 pp; `frac_same_pred(A′, E) = 0.9722`), but Dec A ≫ Dec E by ~23 pp on a1 / b1 — dissociation finding. 2 sign flips vs Dec A (a1 + b1 HMM C1). |

**Chance baseline = 1/36 ≈ 2.8%** for 36-way orientation classification. On R1+R2 the trained decoders A, A′, E, C all sit at 0.53–0.55 (≈ 19× chance); D-raw / D-shape at 0.36–0.37 are lower because the paired-fork focused-only training distribution is out-of-distribution for the 10k HMM eval stream (50/50 focused/routine + ambiguous). Full taxonomy and cross-decoder bias flags: `ARCHITECTURE.md` § "Decoders". Full 17-row matrix with all 7 decoder columns: `results/cross_decoder_comprehensive_with_all_decoders.{json,md}`.

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
