# Decision Ledger

## Active phase-1 contracts

| Decision | Value | Classification | Rationale |
| --- | --- | --- | --- |
| Orientation periodicity | 180 degrees | literature-supported | Orientation, not direction, is the phase-1 stimulus geometry. |
| Core defaults | `O=12`, `T=12` | engineering-only | Keeps the preregistered core run compact while leaving APIs generic. |
| Software geometry | generic in `O` and `T` | engineering-only | Needed for later robustness checks without refactoring. |
| Full trajectory retention | enabled | engineering-only | Time windows are summaries, not the only stored signal. |
| Summary windows | early `1..3`, middle `4..7`, late `8..12` | hypothesis-generating | Useful reporting convention, not a biological claim. |
| Shared normalization | always on in scaffold | engineering-only | Stabilizes the scaffold and remains separate from context-driven gain comparators. |
| Context gain comparator | `context_global_gain` | engineering-only | Avoids conflating scaffold normalization with a tested operator family. |
| Feature-specific operators | `dampening`, `sharpening`, `center_surround` | hypothesis-generating | These are the abstract mechanism families under comparison. |
| Context outputs | next-orientation logits plus optional precision | engineering-only | Restricts model flexibility and prevents hidden shortcut paths. |
| Training objective | predictive only | engineering-only | Held-out assays must not leak into optimization. |
| Prestimulus assays | cue-only or context-only primary, omission secondary | literature-supported | Keeps template readouts distinct from sensory carry-over. |
| Task irrelevance | orthogonal concurrent task | literature-supported | Cleaner than an implicit attention knob for synthetic blocks. |
| Ambiguity | API-ready, not core | engineering-only | Keeps phase 1 narrow without blocking later extensions. |
| Observation pools | `identity`, `gaussian_orientation_bank` | engineering-only | Fixed pooled readouts are needed for readout-dependent comparisons without tuning the pooling model to the result. |
| Energy and homeostasis | implemented, zero-weight by default | hypothesis-generating | The cost terms are part of the broader question, but not part of the first adjudication. |

## Frozen before mechanism comparison

- Adaptation and noise settings are calibrated on neutral data and then held fixed.
- Observation pools are fixed to `identity` and `gaussian_orientation_bank`.
- Generator sanity is split into nuisance-only failure and predictive-structure success checks.
- The same physical transition should flip expectedness across contexts where feasible.
- Learned context outputs are causally masked and limited to prediction plus optional precision.
- Held-out assay metrics are not valid training targets.

## Assumptions in this scaffolding pass

- The config layer uses standard-library dataclasses rather than `pydantic`. This is an implementation choice, not a scientific claim.
- The live tree already includes the generator, model dynamics, context predictor, assay runner, and a full optimizer-driven predictive training loop, so these ledger entries now describe both configuration contracts and implemented components.
- The main orchestration risk is now concentrated in `src/lvc_expectation/runner.py`, not in missing training infrastructure.

## Stage-0 provenance freeze

The repository now treats the following provenance fields as mandatory for new runs:

- `contract_version`
- `artifact_schema_version`
- `metric_schema_version`
- `benchmark_registry_version`
- explicit lineage fields, even when empty

New runs should also emit:

- `resolved_config.json`
- `environment.json`
- `run_fingerprint.json`

These additions freeze provenance only. They do not change the model, scaffold, training objective, or metric semantics.

## Frozen benchmark anchors

The frozen benchmark registry in `src/lvc_expectation/provenance.py` locks the current anchor bundles by run id and bundle fingerprint:

- `phase1_poststim_positive_bundle`
- `phase1_prestim_negative_bundle`
- `phase1_5_decisive_positive_bundle`
- `phase2_source_heldout_negative_bundle`
- `failed_challenger_sweep_bundle_set`
