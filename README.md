# surround_sharpening

A minimal, biologically grounded surround-inhibition mechanism added to a
recurrent predictive orientation circuit — **two config constants, nothing
else** — turning the model's center-boost-only "sharpening" into genuine
flank-suppressed sharpening, and tested for family validity in the energy-
pressured "dampening" regime under the identical architecture.

**Read this first:** [`docs/ARCHITECTURE_AND_SCIENCE.md`](docs/ARCHITECTURE_AND_SCIENCE.md)
— the one document covering the architecture, the mechanism math, the
biological grounding, the training recipe, the full (honest) science story
with every validated number, the measurement conventions, and the provenance
table.

## Layout

```
README.md                        this file
docs/ARCHITECTURE_AND_SCIENCE.md THE document (architecture / mechanism / science)
docs/study_record/               verbatim copies of the study's governing docs
                                 (protocol, design, diagnostics, validation verdict)
src/train_sweep.py               training harness (frozen sweep trainer + the
                                 2-constant surround diff; defaults s=0.05, σ=4.0)
src/simple_net.py                frozen base library (verbatim copy, byte-identical)
src/tuned_emergence_lib.py       frozen tuned-network library (verbatim copy)
src/endpoint_eval_seed.py        official sharpening endpoint evaluator (α=0.0 bars)
src/phase4_reference_eval.py     frozen-artifact reference pinning for the dampening bars
src/phase4_endpoint_eval.py      official dampening endpoint evaluator (α=0.5, P1–P4)
src/ladder_eval_sharpening.py    dose-ladder variant (expected strength as argument)
src/ladder_eval_dampening.py     dose-ladder variant (expected strength as argument)
src/make_flank_sharpening_figs.py  delivered s=0.05 sharpening figures
src/make_family_figs.py            delivered s=0.04 family (both-regime) figures
```

## Quick start

Train one cell (fresh pretrain 3000 steps + one α arm 8000 steps, ~3.5 min on
an RTX 5090; run dirs and checkpoints are written under `--out`, never here):

```bash
cd src
PYTHONHASHSEED=0 python3 -B train_sweep.py \
    --out /path/on/scratch/my_run --seed 8 --alphas 0.0 \
    --recurrent-cell rnn_tanh --device cuda:0
```

Evaluate the endpoint against the pre-registered sharpening bars:

```bash
PYTHONHASHSEED=0 python3 -B endpoint_eval_seed.py \
    /path/on/scratch/my_run/seed_8/alpha_0p0_final.pt report.json
```

Dampening regime: train with `--alphas 0.5` and evaluate with
`phase4_endpoint_eval.py` (P1–P4 bars; reference values pinned from the frozen
original study by `phase4_reference_eval.py`). Any other surround dose: edit
`pred_inhib_strength` in `MODEL_CONFIG` (one line) and use the `ladder_eval_*`
variants, passing the expected strength as the third argument.

Figures: `make_flank_sharpening_figs.py` and `make_family_figs.py` re-render
the delivered figures deterministically from sha-pinned evaluation reports (no
GPU, no training).

## Determinism and reproduction

Everything in the study ran with `PYTHONHASHSEED=0 python3 -B` on `cuda:0`
with deterministic algorithms enabled by the harness. Under those conditions a
(seed, config, device) triple reproduces bitwise — the study's A/A control
reproduced an 11,000-step frozen training run **bitwise** with the surround
strength zeroed (VERDICT.md, Check 3).

## What is NOT in this repository

No checkpoints, no run outputs, no training data (`.gitignore` excludes them).
The evaluators bind to the frozen study assay and frozen reference artifacts
by absolute path on the study machine (reuben-ML) **by design** — they
reproduce the recorded numbers against the immutable study record. Every such
artifact is listed with its sha256 in the provenance table of
`docs/ARCHITECTURE_AND_SCIENCE.md` §7.
