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
| `dampening-analysis` | Re-centered tuning-curve analysis + figures + corrective writeup on the rescue results, including raw/delta/baseline surfaces and paired-state branch-counterfactual probes. |

## Key docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — model architecture, loss table, optional rescue-chain config flags.
- [`RESULTS.md`](RESULTS.md) — original 25-run parameter sweep (3 emergent regimes) + rescue-chain cross-checkpoint summary.
- [`docs/project_summary.md`](docs/project_summary.md) — comprehensive project history (predecessor → simple-dual → Phase 2.x alpha_net → simple-dual-regime → rescue chain).
- [`docs/rescues_1_to_4_summary.md`](docs/rescues_1_to_4_summary.md) — full per-rescue writeup and the 2026-04-13 re-centered analysis update.
- [`docs/training_experiments_analysis_design.md`](docs/training_experiments_analysis_design.md) — baseline training pipeline / paradigm / analysis spec.

## Headline finding (rescue chain)

R4 (DeepTemplate + error-mismatch) exhibits the cleanest Richter (2018) preserved-shape dampening signature: peak −15%, total −19%, FWHM matched within 1.5° between expected and unexpected trials. Main visual: `docs/figures/tuning_ring_recentered_r4.png`. The dampening is **task-state-invariant**, so the preregistered BOTH-regime criterion (focused → Kok sharpening, routine → Richter dampening from one network) is **not** met.

## Dampening-analysis follow-up

The current `dampening-analysis` branch also contains an **aligned pure-R1+2**
follow-up on the paired-state branch-counterfactual surface. This is not a
full all-rescue reanalysis; it is a targeted check of the aligned
`r12_fb24_sharp_050_width_075_rec11_aligned` checkpoint with three readouts:
`raw`, `delta`, and `baseline`.

On the branch-counterfactual **Relevant** surface, the aligned R1+2 checkpoint
now shows:
- `baseline`: expected and unexpected are identical after the baseline-centering
  fix (`peak=0.375691`, `FWHM=39.389691°` for both).
- `raw`: expected is lower and slightly narrower than unexpected
  (`peak 0.449558 vs 0.507532`, `FWHM 33.237447° vs 33.515513°`).
- `delta`: expected is much lower and much narrower than unexpected
  (`peak 0.072438 vs 0.491614`, `FWHM 27.581737° vs 45.064195°`).

The baseline-centering bug was analysis-only: identical pre-probe baselines
were previously re-centered by different branch probe channels, which made the
same baseline appear artificially opposite. Baseline mode now uses the shared
predicted/expected channel for both branches.

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
