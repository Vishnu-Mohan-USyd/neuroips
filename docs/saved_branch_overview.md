# Saved Branch Overview

This document summarizes the saved experimental branch at:

- Local path: `/home/vishnu/coding_proj/codex_exp/neuroips_publish_strict_shared`
- Git branch: `publish/strict-shared-no-pred-conf-results`
- Saved experiment base commit: `b70de8b`

This is a consolidated orientation note for the branch. It does not replace the
source files, the per-phase result notes, or the validation record in
`docs/expectation_energy_repair.md`.

## Branch Purpose

The branch preserves a strict shared-mechanism version of the SOM/VIP
expectation model. It keeps sharpen/attend and dampen/save effects as outcomes
of training pressure and validation, not as explicit shape templates or
expected-vs-unexpected response targets.

The saved branch has two layers of history:

- The original Phase A and Phase B SOM/VIP story in `README.md`,
  `phaseA_somvip/`, and `phaseB_somvip/`.
- The later expectation-energy repair tools and validation note in `tools/`
  and `docs/expectation_energy_repair.md`.

The repair artifacts were generated outside the repository tree and the
canonical repository checkpoints were not overwritten.

## Code And Architecture

The shared core model is `simple_net.py`.

The core network has:

- A fixed L4 orientation ring with `N=36` channels at 5 degrees per channel.
- A trainable L4-to-L2/3 feedforward map, `W_ff`.
- An L2/3 representation that receives feedforward drive plus top-down
  prediction feedback.
- A GRU recurrent predictor that reads L2/3 and projects the next-step
  prediction back down through `W_fb`.
- A built-in L2/3 decoder, `net.decoder`, used as the current-orientation
  decoder in the strict repair validators.

The headline path uses `SimpleNet(use_circuit=True)`, which routes top-down
feedback through a Dale-compliant SOM/VIP microcircuit. The learned circuit
gains are nonnegative through `softplus`; inhibitory effects come from fixed
structural minus signs, not negative learned weights. In the circuit path,
top-down feedback can drive VIP, SOM, and pyramidal excitation; VIP inhibits
SOM, and SOM inhibits L2/3 pyramidal activity.

`SimpleNet(use_circuit=True, context=True)` adds one context gain,
`g_ctx_raw`, into the VIP drive. In the combined Phase B repair, context
`+1` means attend and context `-1` means save. The context path does not add a
separate readout contract; the repaired validator requires the built-in
`net.decoder`.

`simple_net.py` also contains optional local competition, local inhibition,
feedback-gated inhibition, SOM feedback pooling, topographic SOM/VIP routing,
and prediction-error-style terms. These are default-off mechanisms unless a
driver or checkpoint config explicitly enables them. Alternative non-headline
feedback modes, such as additive, subtractive, and signed feedback variants,
are retained for lineage but are not the strict repair's main validation
contract.

The strict repair scripts are:

- `tools/train_energy_repair.py`: trains separate independent sharpen and
  dampen SOM/VIP checkpoints from a shared phase-1 initialization and writes
  them only to an explicit external output directory.
- `tools/validate_energy_decoding_shape.py`: validates the independent
  sharpen and dampen checkpoints on deterministic expected-vs-orthogonal
  paired sequences.
- `tools/train_ctx_energy_repair.py`: trains one context-switching checkpoint
  with `SimpleNet(use_circuit=True, context=True)`, saving a Phase B-style
  wrapper dict containing `net`, `read`, and `cfg`.
- `tools/validate_ctx_energy_decoding_shape.py`: validates the combined
  context checkpoint, including clean and noisy decoder checks, cross-context
  checks, and Phase B legacy-equivalent controls.

The tuned-emergence tools in `tools/tuned_emergence_lib.py` and related
scripts are separate experimental lineage. They use a fixed orientation-tuned
L2/3 basis and constrained readouts. They are not the primary strict repair
validation path documented in `docs/expectation_energy_repair.md`.

## Independent Vs Combined Models

The independent repair uses two separate checkpoints:

- Sharpen checkpoint: `SimpleNet(use_circuit=True)`
- Dampen checkpoint: `SimpleNet(use_circuit=True)`

These two models share the same architecture and start from the same phase-1
style representation setup, but they are trained separately under different
objective pressure. The distinction is an objective/regime distinction, not a
hardcoded shape distinction. The validator loads each checkpoint into
`SimpleNet(use_circuit=True)` and checks the two outcomes separately.

The combined repair uses one checkpoint:

- Combined checkpoint: `SimpleNet(use_circuit=True, context=True)`

The combined model has one set of weights and switches at runtime through the
context input. Context `+1` is attend and context `-1` is save. The combined
validator verifies both contexts in the same network, checks that save uses
less expected energy than attend, and includes legacy-equivalent controls for
feedback-off floor parity and `g_ctx` lesion collapse.

The important difference is therefore:

- Independent repair: two separately trained networks with the same circuit
  architecture.
- Combined repair: one context-gated network with shared weights and one
  runtime context input.

In both cases, validation uses paired sequences with `K=4`, 36 starts, and
velocities `[-3, -2, -1, 1, 2, 3]`, comparing each expected continuation with a
90-degree orthogonal unexpected continuation sharing the same prefix.

## Strict Constraints From MEMORY.md

`MEMORY.md` is part of the branch contract. The main constraints are:

- Response shape is a validation outcome only.
- Training objectives must not include explicit dampen/sharpen shape losses,
  center/flank objectives, local annulus terms, or target curve templates.
- Expected-vs-unexpected energy and shape comparisons are validation outcomes
  only.
- Training must not include expected-vs-unexpected contrast losses that know a
  stimulus is expected or unexpected by construction.
- Energy pressure must be general metabolic or homeostatic pressure across the
  relevant circumstances, not a selective expected-case penalty.
- Regime-specific circuit or gain parameters that directly make feedback more
  suppressive or less suppressive are disallowed.
- In particular, regime-specific `pred_feature_supp_strength` is disallowed.
- Sharpen and dampen may share architecture, initial form, and circuit
  parameters. Learned weights may diverge only because training objective
  pressures differ.
- Allowed regime differences are objective weights such as energy, task,
  precision, and homeostasis weights.

The practical interpretation for future edits is that a passing result is not
enough if the mechanism was produced by putting the validation answer directly
into the loss or into regime-specific circuit knobs.

## Validation Results Already Documented

The recorded strict repair validation is in
`docs/expectation_energy_repair.md`.

That note documents:

- The exact independent validation command:
  `python tools/validate_energy_decoding_shape.py --sharpen ... --dampen ...`
- The exact combined validation command:
  `python tools/validate_ctx_energy_decoding_shape.py --ckpt ...`
- Local checkpoint and log artifact paths.
- The deterministic paired validation set of 216 expected-vs-orthogonal pairs.
- A primary validation table showing expected energy below unexpected energy
  in independent sharpen, independent dampen, combined attend, and combined
  save.
- Held-out next-step prediction accuracy above the documented threshold in all
  listed cases.
- Shape pass results: sharpen/attend center enhanced; dampen/save center
  suppressed below feedback-off floor.
- Noisy stress checks at Gaussian noise `sigma=1.0`.

The primary table in that note reports:

| Model/state | Expected energy | Unexpected energy | Expected CE | Unexpected CE | Held acc | Shape |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Independent sharpen | 1.6351 | 1.7116 | 0.0215 | 0.4508 | 82.03% | pass, center enhanced |
| Independent dampen | 1.1112 | 1.3087 | 0.1872 | 0.0676 | 81.74% | pass, center suppressed below floor |
| Combined ctx=+1 attend | 1.7634 | 1.8175 | 0.0216 | 0.0737 | 81.48% | pass, center enhanced |
| Combined ctx=-1 save | 1.5535 | 1.5928 | 0.0560 | 0.0342 | 81.21% | pass, center suppressed below floor |

The same note records the noisy stress summary:

- Independent expected accuracy at `sigma=1.0`: sharpen 80.96%, dampen 64.01%.
- Combined expected accuracy at `sigma=1.0`: attend 90.7%, save 82.8%.
- Combined expected CE at `sigma=1.0`: attend 0.243, save 0.400.

These results support the intended tradeoff recorded there: expected stimuli
are cheaper in both regimes, while attend/sharpen preserves more precise
expected decoding than save/dampen.

## Artifact And Scope Caveats

This is an experimental saved branch, not a polished release artifact.

Important caveats:

- The repaired checkpoints, logs, summaries, and JSON reports referenced by
  `docs/expectation_energy_repair.md` are local generated artifacts outside
  this repository.
- The canonical repository checkpoints were not overwritten by the repair
  scripts.
- Anyone reproducing the validation must either keep those local artifact paths
  available or regenerate equivalent artifacts with the documented training
  scripts.
- The validation covers the specified paired-sequence regime and the documented
  Phase B legacy-equivalent controls. Broader stimulus distributions and
  additional seeds should be validated separately before making broader claims.
- The repair objectives are differentiable engineering approximations of
  prediction-weighted metabolic pressure. They should not be described as a
  fully biological implementation without additional evidence.
- Future changes should preserve the strict MEMORY.md boundary: validation can
  ask whether expected and unexpected responses differ, but training should not
  be handed that comparison as the answer.
