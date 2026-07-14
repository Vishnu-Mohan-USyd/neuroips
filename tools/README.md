# Tool map

This directory contains one canonical current workflow and several retained
development/repair workflows. Choose tools by section; similarly named scripts
are not interchangeable.

## Current task–energy workflow

Run these from the repository root in this order:

1. `train_emergent_task_energy_axis.py` — train one seed-specific common task
   pretrain and fixed-step alpha arms under
   `J(alpha)=(1-alpha)T+alpha E`.
2. `assay_emergent_task_energy_axis.py` — replay the fixed 216 operational
   continuation-A / OOD-reversal-B pairs and record energy, decoding, and shape
   measurements; this tool measures but does not accept an endpoint.
3. [`evaluate_emergent_task_energy_gates.py`](evaluate_emergent_task_energy_gates.py)
   — portable post-hoc evaluator for the frozen three-family gate specification.
   Its sealed CUDA replay on seeds `8–11` passed and matched the authoritative
   result exactly; it reproduces gate decisions, not the historical external
   assay ledger or final seal. See the
   [10 focused tests](../tests/test_evaluate_emergent_task_energy_gates.py) and
   [selection-record schema/binding tests](../tests/test_endpoint_selection_record_schema.py).
4. `plot_emergent_reference_figures.py` — replay selected checkpoints and write
   the two tuning panels, decoding bars, phase-space panel, and `plot_data.json`;
   plotting is presentation, not endpoint acceptance.

Shared current library:

- `tuned_emergence_lib.py` — defines `SimpleTunedNet`, the fixed local
  orientation basis, recurrent predictor, posterior-over-prior feedback
  transform, SOM/VIP-inspired sign-constrained rate motif, and causal unroll.

The full commands, including four-seed run directories, are in the
[canonical guide](../docs/emergent_task_energy_axis.md#rtx-5090-reproduction-recipes).

### Model boundary

The current workflow instantiates
`tools.tuned_emergence_lib.SimpleTunedNet`. Root
`simple_net.SimpleNet` is the legacy Phase A/B class and is **not** the current
architecture. `tuned_emergence_lib.py` imports shared orientation constants,
the fixed L4 code, and sequence-generation utilities from `simple_net.py`; that
utility import does not make the two model classes equivalent.

## Retained six-alpha summary tool

- `aggregate_emergent_task_energy_assays.py` — aggregates already-written
  seeds `0–3` six-alpha assay JSON into the historical compact scalar summary.
  It neither replays checkpoints nor evaluates the selected `.5` endpoint.

## Earlier tuned-basis exploration

These scripts predate the selected task–energy protocol and are retained for
development history:

- `train_tuned_emergence.py` — earlier fixed-basis tuned-network trainer.
- `validate_tuned_emergence.py` — held-out validator paired with that trainer.
- `plot_tuned_raw_tuning.py` — raw aligned tuning plots for its checkpoints.

## Natural-emergence and tuning-shape development

These are superseded exploratory branches, not alternate entry points for the
current result:

- `train_natural_emergence.py` — earlier ordinary-sequence task/precision/rate
  objective exploration.
- `train_tuning_shape_natural.py` — earlier local/sparse feedforward and
  naturalistic rate-demand experiment.
- `train_tuning_shape_repair.py` — additive tuning-shape repair experiment with
  objectives that are not used by the current workflow.
- `validate_independent_tuning_shape_strict.py` — strict validator for those
  independent repaired checkpoints.

## Legacy Phase A/B repair tools

These scripts operate on the legacy `SimpleNet` lineage and its checkpoint
formats:

- `train_energy_repair.py` — additive Phase A energy/decoding repair trainer.
- `validate_energy_decoding_shape.py` — paired validator for repaired Phase A
  checkpoints.
- `train_ctx_energy_repair.py` — context-switching Phase B repair trainer.
- `validate_ctx_energy_decoding_shape.py` — validator for repaired Phase B
  context checkpoints.

For the original Phase A and Phase B experiments themselves, use
[`phaseA_somvip/RESULTS.md`](../phaseA_somvip/RESULTS.md) and
[`phaseB_somvip/RESULTS.md`](../phaseB_somvip/RESULTS.md). Do not feed legacy
checkpoints to the current assay or combine legacy objectives with the current
alpha-axis interpretation.
