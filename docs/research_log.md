# Research Log

## 2026-03-31

### Phase-1 framing

The repository is effectively greenfield apart from `paper_3_litrev_v1.pdf`. The design rationale for phase 1 therefore comes from the literature-review PDF plus the external discussion that narrowed the target: do not start with a heavy biological simulator, and do not try to solve "predictive coding" in general. Start with a synthetic, mechanism-identifiable orientation scaffold that is strict about confounds and about what is and is not optimized.

The central scientific framing is not "find one universal winner." The current literature does not support that. The codebase should instead support a mapping from operator family to assay, readout, and time window on a shared scaffold.

### Why the scaffold is small

The phase-1 target is to compare abstract operator families under tightly controlled conditions:

- `adaptation_only`
- `context_global_gain`
- `angular_dampening`
- `angular_sharpening`
- `angular_center_surround`

That comparison motivates generic orientation geometry, explicit adaptation, separate always-on scaffold normalization, a context-driven operator family, and held-out assays. It does not justify phase-1 claims about explicit cell types, full predictive-coding microcircuits, or STDP-driven learning.

This is why the defaults stay compact: `O=12`, `T=12`, full trajectories retained, and fixed early/middle/late summary windows derived from the stored trajectories rather than replacing them.

### Why context is restricted

The context pathway is causally masked and limited to a predicted next-orientation distribution plus an optional scalar precision signal. This is an engineering discipline choice to keep the model falsifiable and to avoid hidden shortcut paths from context into the readout. The training contract is therefore predictive-only: held-out assay metrics do not belong in the optimization target.

### Why cue-only prestimulus assays are primary

Prestimulus cue-only or context-only template assays provide a cleaner test of template formation than omission alone because they separate prediction from sensory rebound or carry-over. Omission remains in scope as a secondary check because the literature also treats omitted expected input as informative.

### Why the geometry is strict

Orientation is treated as a 180-degree periodic circular variable from day one. This is not a cosmetic choice. The symmetry checks, kernel parameterization, and robustness sweeps all depend on using orientation rather than direction geometry. The software contract therefore rejects non-180-degree periodicity in phase 1.

### Why the generator contract is strict

Expectedness should arise from conditional structure, not from nuisance imbalance. The generator and later validation must therefore preserve the intended predictive variables while separately checking that nuisance-only metadata cannot decode condition above chance. The live tree now includes a paradigm generator plus sanity-check utilities for nuisance-only failure and predictive-structure success, so this requirement is no longer only aspirational documentation.

### What remains hypothesis-generating

The exact mapping from the abstract operator family to biological interneuron classes remains unresolved. The exact number and placement of temporal summary windows also remains a reporting convention rather than a settled biological claim. Likewise, the exact contribution of energy optimization to expectation-like effects is left open in phase 1: the codebase should log those terms and support later sweeps, but not use them to define the first adjudication.

### Current implementation note

The live tree now includes the phase-1 paradigm generator, scaffold/model dynamics, context predictor, held-out assay runner, and a full optimizer-driven predictive training loop in `src/lvc_expectation/train.py`. The runner layer in `src/lvc_expectation/runner.py` now orchestrates tranche runs, prestim gates, local-global probe gates, alignment gates, and opt-in recovery schedules. The remaining engineering caveat is provenance and orchestration complexity, not the absence of a training stack.

## 2026-04-01

### Stage-0 provenance freeze

The repository now treats provenance as a first-class requirement because git history is not a reliable implementation record. New runs should emit:

- versioned manifest metadata
- resolved config snapshot
- environment snapshot
- run fingerprint

The codebase also freezes a small benchmark registry for current bundle anchors:

- phase-1 post-stim positive bundle
- phase-1 prestim negative bundle
- phase-1.5 decisive positive bundle
- phase-2 source-heldout negative bundle
- failed challenger sweep bundle set

This is a reporting and reproducibility freeze only. It does not alter model dynamics, training objectives, or metric semantics.
