# Emergent sharpening and dampening on a task–energy axis

This repository tests whether one recurrent orientation-circuit architecture
can express sharpening-like or dampening-like activity when only the balance
between task performance and low neural activity changes.

The current experiment is the
[task–energy axis workflow](docs/emergent_task_energy_axis.md). Older Phase A,
Phase B, and repair experiments remain available as scientific history, but
they are not the current model or evidence base.

## Scientific question

Can a circuit learn two response regimes without a loss that names an expected
stimulus or prescribes a response shape?

- With greater task pressure and little rate pressure, the representation
  should be sharpening-like: operational continuation A decodes better than
  matched operational OOD reversal B, with center enhancement and flank
  suppression relative to the first-stimulus response.
- With balanced task and rate pressure, it should be dampening-like: A uses
  less activity and decodes worse than B, while its center is suppressed more
  strongly than its flanks.

These are assay outcomes. They are not training labels or target curves.

## Current architecture

```text
orientation
  → fixed 36-channel L4 code
  → fixed local nonnegative L4→L2/3 basis
  → 36-channel nonnegative L2/3 rates
  → 64-unit GRU predictor → W_fb
  → posterior-over-prior feedback evidence one step later
  → SOM/VIP-inspired Dale-sign-constrained rate motif
  → fixed local divisive competition
```

The executable model class is `tools.tuned_emergence_lib.SimpleTunedNet`.
Feedback changes through learned weights and nonnegative motif gains; the
architecture and sign structure are identical across objective arms.

> **Module boundary:** root `simple_net.SimpleNet` is the legacy Phase A/B
> model. The current workflow does **not** instantiate it. `SimpleTunedNet`
> imports only shared orientation constants, L4 coding, and sequence utilities
> from `simple_net.py`. Do not infer the current architecture from the legacy
> `SimpleNet` class or its module docstring.

## Objective: one equation, two terms

Every arm minimizes

`J(alpha) = (1 - alpha) T + alpha E`,

where `T` averages normalized next-step prediction and noisy
current-orientation precision, and `E=mean(r)/R_ref` is a normalized L2/3
mean-rate proxy. `E` is an engineering proxy, not ATP, oxygen consumption, or a
whole-brain energy measurement.

Training never sees the post-hoc A/B assay labels, center/flank windows,
literal-`t0` baseline, acceptance thresholds, or a desired tuning curve.
`alpha=0.5` is simply 50% task and 50% rate-proxy pressure—not a new loss or a
regime-specific intervention.

## Selected result

`alpha=0.5` was selected after development and then confirmed from scratch on
independent seeds `8–11`. It passed the three scientific validation families:

1. **Energy:** final mean L2/3 rate was lower for continuation A than reversal B
   in every seed.
2. **Decoding:** one condition-blind, noise-held-out 36-class decoder was less
   accurate for A than B, while both remained above chance, in every seed.
3. **Shape:** A had center suppression with **relative flank sparing** against B
   and the first-stimulus response. Both center and flanks remained below that
   baseline; “relative” does not mean absolute flank enhancement.

All `48/48` per-seed checks implementing those three families passed. Mean
whole-profile retention was `M=0.29913366797987234`, above the fixed `.250`
cohort floor. The machine-readable
[endpoint selection record](figures/emergent_reference_comparison/endpoint_selection_record.json)
preserves the accepted `.5` result and the rejected `.6` calibration path.

The task-only `alpha=0.0` arm supplies the sharpening-like comparator in the
current plots. It uses the same architecture and `J(alpha)`; only `alpha`
changes. The plotted comparison uses matched seeds `8–11` and endpoints `.0`
and `.5`.

### Literal first-stimulus baseline

The gray tuning curve is the same arm's ordinary response at the first sequence
step, with hidden state, prior feedback, and adaptation state equal to zero. It
is not reversal B and not a feedback-off final response.

## Quick start

Python 3.10+ and PyTorch are required; an NVIDIA GPU is recommended for
training.

```bash
python -m pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
RUN_ROOT="${RUN_ROOT:-$HOME/neuroips_runs/task_energy_quickstart}"

python -B tools/train_emergent_task_energy_axis.py \
  --seed 0 \
  --device cuda:0 \
  --out "$RUN_ROOT" \
  --alphas 0.0 0.5 0.9 \
  --freeze-local-comp \
  --feedback-mode posterior_prior_excess

python -B tools/assay_emergent_task_energy_axis.py \
  --run-dir "$RUN_ROOT/seed_0" \
  --device cuda:0 \
  --out "$RUN_ROOT/seed_0/endpoint_assay.json" \
  --alphas 0.0 0.5 0.9
```

That is a single-seed run, not reproduction of the confirmed cohort. Use the
[RTX 5090 recipes](docs/emergent_task_energy_axis.md#rtx-5090-reproduction-recipes)
for the complete four-seed train, assay, gate, and figure workflow. Tool order
and historical-script boundaries are listed in [tools/README.md](tools/README.md).

## Repository map

| Path | Purpose |
| --- | --- |
| [`docs/emergent_task_energy_axis.md`](docs/emergent_task_energy_axis.md) | Canonical architecture, exact losses, assay, results, provenance, limitations, and reproduction |
| [`tools/`](tools/) | Current trainer/assay/plot pipeline plus clearly labeled historical scripts; see [`tools/README.md`](tools/README.md) |
| [`figures/emergent_reference_comparison/`](figures/emergent_reference_comparison/) | Current `.0` versus `.5` figures, plotted values, and endpoint selection history; see its [`README.md`](figures/emergent_reference_comparison/README.md) |
| [`docs/research_log.md`](docs/research_log.md) | Chronological discoveries and biological interpretation notes |
| [`phaseA_somvip/RESULTS.md`](phaseA_somvip/RESULTS.md) | Legacy objective-grid Phase A result |
| [`phaseB_somvip/RESULTS.md`](phaseB_somvip/RESULTS.md) | Legacy runtime-context Phase B result |
| [`simple_net.py`](simple_net.py) | Legacy `SimpleNet` plus orientation utilities reused by the current tuned library |
| [`requirements.txt`](requirements.txt) | Runtime dependencies |

## Current versus legacy

Current scientific claims come from `SimpleTunedNet`,
`train_emergent_task_energy_axis.py`, the fixed continuation/reversal assay,
the selected `alpha=.5` confirmation, and the artifacts under
`figures/emergent_reference_comparison/`.

The Phase A/B directories and the remaining repair/tuning scripts are retained
for provenance and earlier scientific hypotheses. Their checkpoints, context
switches, feedback-off floors, and regime-specific objectives must not be mixed
with the current experiment. Start with the canonical guide, then follow legacy
links only when studying project history.
