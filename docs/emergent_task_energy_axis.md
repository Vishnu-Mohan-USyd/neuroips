# Current task–energy axis workflow

This is the canonical guide to the repository's current task–energy experiment.
It separates the exploratory six-alpha sweep, endpoint calibration, and
independent four-seed confirmation so that development observations are not
mistaken for fresh evidence. The older Phase A/Phase B and repair notes remain
historical lineage; they do not define the model or confirmation described here.

## Newcomer overview

The experiment asks what one recurrent orientation circuit learns when the same
task objective is traded against a **normalized L2/3 mean-rate proxy**. Every
optimization arm starts from the same seed-specific task pretrain and uses the
same architecture. The only arm coordinate is

`J(alpha) = (1 - alpha) T + alpha E`,

with no additional objective at any endpoint. Training never receives the
post-hoc continuation/reversal labels, matched assay pairs, endpoint names,
center/flank windows, amplitude thresholds, or target response shapes.

The task-only **`alpha=0.0`** endpoint is the sharpening-like comparator. The
current dampening endpoint is the balanced **`alpha=0.5`** coordinate: 50% of
the normalized task term and 50% of the normalized mean-rate term. It is not a
new loss, fitted amplitude target, or regime-specific circuit. On independent
fresh seeds `8,9,10,11`, `alpha=0.5` passed all three scientific validation
families:

- operational continuation A used less final mean L2/3 activity than matched
  operational OOD reversal B;
- one condition-blind noise-held-out decoder was less accurate for A than B,
  while both remained above 36-way chance;
- the aligned A profile had a suppressed center with relatively spared flanks
  against both B and the literal first-stimulus baseline.

Here **relative flank sparing** means that flanks retain a larger fraction of
their `t=0` baseline than the center. It does not claim that absolute flank
activity exceeds baseline. These are post-training measurements, not labels
supplied to the loss.

The retained seeds `0–3` task-only comparator supplies the other regime: it
kept A below B in mean activity
(`(B-A)/(B+epsilon_rate)=.0761`, where
`epsilon_rate=1e-8*N*R_ref`), decoded A better than B (`.9975` versus `.3439`),
raised the center by `.279851 AU` from `t=0`, and changed the flanks by only
`-.012470 AU`. These are exploratory-cohort values; the endpoint plotter
replays the selected checkpoints rather than fitting a target curve.

The endpoint was chosen through a transparent calibration path. `alpha=0.6`
passed the development screen on seeds `0–3`, but on fresh seeds `4–7` it
failed amplitude retention. Its stored assay leaves still passed energy,
decoding, `dC<dF`, and `dQ<0`; no stored `Cret/Fret` claim is made for that
cohort. It was rejected. `alpha=0.5` then passed development seeds `4–7` and a
separate from-scratch confirmation on seeds `8–11`. The original seeds `0–3`
six-alpha sweep is preserved below as exploratory lineage rather than recast as
confirmation. The portable
[`endpoint_selection_record.json`](../figures/emergent_reference_comparison/endpoint_selection_record.json)
is the machine-readable selection history.

## Newcomer reading path

For a first pass, read these sections in order:

1. [Architecture, signs, and timing](#architecture-signs-and-timing).
2. [Training objective](#training-objective).
3. [Operational assay](#operational-continuationreversal-assay) and its
   [literal first-stimulus baseline](#literal-first-stimulus-tuning-baseline).
4. [Calibration and fresh confirmation](#dampening-calibration-and-fresh-confirmation).
5. [Reproduction recipes](#rtx-5090-reproduction-recipes) and
   [limitations](#limitations).

The current workflow is compact even though its audit record is detailed:

```text
ordinary momentum sequences
  → common task pretrain for one seed
  → identical architecture cloned into alpha arms
  → J(alpha)=(1-alpha)T+alpha E
  → fixed 216-pair continuation-A / OOD-reversal-B assay
  → energy + decoding + aligned-shape readout families
  → endpoint selection record
  → checkpoint-replayed figures
```

## Artifact provenance

Executable code and checkpoint contents define the computation. Three result
sets must be kept distinct:

```text
(A) historical seeds 0–3, six-alpha checkpoints
    └─ tools/assay_emergent_task_energy_axis.py replay
       └─ per-seed endpoint_assay.json audit records
          └─ tools/aggregate_emergent_task_energy_assays.py
             └─ figures/emergent_reference_comparison/all_alpha_assay_summary.json
                └─ historical six-alpha table in this guide

(B) selected alpha=0.0 sharpening and alpha=0.5 dampening checkpoints
    └─ tools/plot_emergent_reference_figures.py direct replay
       └─ in-memory aggregates
          ├─ figures/emergent_reference_comparison/plot_data.json
          └─ four sibling PNG figures

(C) fresh seeds 8–11, alpha=0.5 and alpha=0.9 checkpoints
    └─ fixed 216-pair assay and literal-t0 replay
       └─ corrected frozen gate evaluation
          └─ 32-entry assay ledger
             └─ final all-assays seal
```

Branch C is the scientific confirmation. Its external run directory is not
committed; identify it portably by final seal SHA-256
`027feb665537e1f54628e9e7af1ff5b25bdb759e067ff02e6b751fb42e37cd51`
and assay-ledger SHA-256
`04404bd8efdaba8a506b686d746c79bbb03b4212799ced43fd3c8ef2c3fb77a4`.
The 58-entry training ledger is
`f248a263ea285cce5e0ad16db2fb95a357cee2c1705a35f66b9f6e6eae53b32b`
and its training seal is
`2de0c984c0346f39e4bf82aebad814bc76b03057b00ecf432819164669ea557b`.
The tracked
[`endpoint_selection_record.json`](../figures/emergent_reference_comparison/endpoint_selection_record.json)
condenses the `.6` rejection and `.5` selection into logical run identifiers
and hashes; it does not replace the external sealed evidence.

Per-seed `training_summary.json` files record configuration, state hashes, and
training diagnostics. They are not measurement or plotter inputs; the compact
six-alpha aggregator reads them only to corroborate protocol and checkpoint
provenance. Its numerical measurements come from `endpoint_assay.json`. The
plotter directly remeasures the two displayed checkpoints, writes
`plot_data.json` and the PNGs as sibling outputs. Visual resemblance is not a
validation decision; branch C's three scientific readout families are.

The compact all-alpha artifact aggregates the 14 scalar fields already present
in each standalone assay. Those assay files do not contain raw aligned
profiles, so the aggregator explicitly does not reconstruct them. The 36-bin
A/B and first-stimulus curves in `plot_data.json` exist only for the two
displayed endpoints and belong to provenance branch B.

Its generator provenance distinguishes repository context from executable
source identity:

- `repository_base_commit` is the commit checked out beneath the worktree; it
  does not assert that the commit contains the generator file;
- `repository_worktree_dirty_at_generation` records whether tracked or
  untracked worktree changes were present when the artifact was generated;
- `source_file_sha256` hashes the exact
  `tools/aggregate_emergent_task_energy_assays.py` bytes that executed.

This split is necessary whenever the generator is new or modified relative to
the checked-out base. The source-file hash, not the base commit alone, identifies
the executed generator. In the tracked aggregate,
`repository_worktree_dirty_at_generation` is `true`; its base commit is
repository context, not a clean-snapshot guarantee.

## File map

| Path | Role |
| --- | --- |
| [`tools/README.md`](../tools/README.md) | Current execution order and explicit boundary between canonical, repair, and legacy scripts |
| [`tools/tuned_emergence_lib.py`](../tools/tuned_emergence_lib.py) | Fixed orientation basis, L2/3 rate circuit, recurrent predictor, feedback transform, and causal unroll timing |
| [`tools/train_emergent_task_energy_axis.py`](../tools/train_emergent_task_energy_axis.py) | Common task pretrain, configured task–energy alpha arms, optimizer policy, deterministic streams, and checkpoints |
| [`tools/assay_emergent_task_energy_axis.py`](../tools/assay_emergent_task_energy_axis.py) | Fixed 216-pair operational continuation/reversal assay and its three readout families |
| [`tools/evaluate_emergent_task_energy_gates.py`](../tools/evaluate_emergent_task_energy_gates.py) | Validated portable frozen-gate replay; reproduces gate decisions but does not regenerate the historical external ledger or seal. Tests: [evaluator](../tests/test_evaluate_emergent_task_energy_gates.py), [schema/bindings](../tests/test_endpoint_selection_record_schema.py) |
| [`tools/aggregate_emergent_task_energy_assays.py`](../tools/aggregate_emergent_task_energy_assays.py) | Historical seeds `0–3` six-alpha scalar summary; not the selected-endpoint evaluator |
| [`tools/plot_emergent_reference_figures.py`](../tools/plot_emergent_reference_figures.py) | Four-seed checkpoint replay, literal first-stimulus tuning baseline, seed aggregation, JSON, and reference-layout figures |
| [`figures/emergent_reference_comparison/README.md`](../figures/emergent_reference_comparison/README.md) | Artifact-by-artifact interpretation of the current PNG and JSON bundle |
| [`figures/emergent_reference_comparison/endpoint_selection_record.json`](../figures/emergent_reference_comparison/endpoint_selection_record.json) | Portable `.6` rejection and `.5` selection history with frozen gates and evidence hashes |
| [`figures/emergent_reference_comparison/all_alpha_assay_summary.json`](../figures/emergent_reference_comparison/all_alpha_assay_summary.json) | Historical seeds `0–3`: compact copy of 14 existing assay metrics for every seed×alpha, plus mean/sample SEM and provenance |
| [`figures/emergent_reference_comparison/plot_data.json`](../figures/emergent_reference_comparison/plot_data.json) | Machine-readable values used by the current endpoint figures |
| [`figures/emergent_reference_comparison/`](../figures/emergent_reference_comparison/) | Two tuning panels, grouped decoding bars, and decoding/rate phase space |

> **Model-module boundary:** the current tools instantiate
> `tools.tuned_emergence_lib.SimpleTunedNet`. Root `simple_net.SimpleNet` is the
> legacy Phase A/B model. The tuned library imports shared orientation constants,
> L4 coding, and sequence utilities from `simple_net.py`; it does not instantiate
> or inherit the legacy class.

## Architecture, signs, and timing

### Orientation representation

- `N=36` nominal orientation channels tile a 180-degree ring at 5 nominal
  degrees per channel. L4 uses a fixed circular Gaussian with 12-degree width.
- L4-to-L2/3 is a fixed, nonnegative, row-normalized circular Gaussian with
  width `1.1` channels (5.5 nominal degrees) and gain `1.6`. It is not the
  trainable dense map described in the legacy README.
- L2/3 is a 36-channel nonnegative rate population in arbitrary activity units
  (AU). Its active divisive local pool is a row-normalized circular Gaussian
  with width `2.0` channels (10 nominal degrees), power `1.0`, and strength
  `ln(2)`.
- A 64-unit `GRUCell` and a biased linear `W_fb: 64 -> 36` form the abstract
  temporal predictor. They carry no calibrated biological time or rate units.

### Feedback evidence and rate motif

Let `z_t` be the predictor logits available after processing time `t`. The next
step receives posterior-over-uniform-prior excess evidence

`f_{t+1} = ReLU(36 softmax(z_t) - 1)`.

This transform is nonnegative and invariant to a common logit shift. The raw
logits used by next-step cross entropy are not replaced by the transform.

The model uses a **Dale-sign-constrained rate motif**, not a conductance model or
an identified cortical microcircuit. With nonnegative gains
`g=(g0,g1,g2,g3,g4)=softplus(circ_raw)`, feedforward drive `d`, and feedback
evidence `f`, its active terms are

`vip = ReLU(g0 f)`,

`som = ReLU(g1 f - g2 vip)`,

`u = ReLU(d + g3 f - g4 som)`.

The configured predictive inhibition, feature suppression, adaptation, and
rate-saturation auxiliaries are zero. Under those zero auxiliaries and
nonnegative `f`, the motif reduces channelwise to

`u_i = ReLU(d_i + k f_i)`,

`k = g3 - g4 ReLU(g1 - g2 g0)`.

The reported rate then applies fixed local divisive competition,

`r = u / (1 + lambda K u)`,

where `K` is the fixed row-normalized local-pooling matrix and
`lambda=ln(2)`. This scalar reduction explains only the instantaneous sign
structure. The GRU and `W_fb` retrain in every arm, so between-arm behavior
cannot be attributed to the five motif gains alone.

### Causal timing

For a batch of sequences `theta` with shape `[B,S]`, the unroll order at each
abstract time step is:

1. compute L2/3 from the current L4 code, prior `pred_down`, and prior
   adaptation state;
2. update adaptation from that L2/3 response;
3. update the GRU from that response;
4. compute raw `W_fb` logits;
5. transform those logits into `pred_down` for the next time step.

At `t=0`, hidden state, feedback state, and adaptation state are exactly zero.
There is no pre-stimulus evaluation of `W_fb(h0)`. Time is an abstract sequence
index; the model claims no membrane time constant, milliseconds, or conduction
delay.

## What is fixed and what trains

| Component | Executed value | Common 3,000-step task pretrain | 8,000-step alpha arm |
| --- | --- | --- | --- |
| L4 and L4-to-L2/3 basis | 36 channels; L4 width 12 degrees; feedforward width 1.1 channels and gain 1.6 | Fixed buffers | Fixed buffers |
| GRU and `W_fb` | 64 hidden units; `W_fb` maps 64 to 36 with bias | Trainable | Trainable from the identical common state |
| Five `circ_raw` motif gains | `softplus(0)=ln(2)` at construction | Frozen | Trainable from the identical common state |
| Local divisive competition | strength `ln(2)`, width 2 channels, power 1 | Parameter exists but is frozen by the optimizer policy | Frozen and byte-checked with `--freeze-local-comp` |
| Built-in `decode()` and gain | population-vector mode, normalized, gain 8 | Frozen and not called by the executed loss | Frozen and unused by trainer, assay, and plotter |
| Predictive-inhibition auxiliary | strength 0; stored width 0.65 channels | Inert | Inert |
| Feature-suppression auxiliary | strength 0 | Inert | Inert |
| Adaptation auxiliary | strength 0; stored decay 0.85 and width 1 channel | Inert | Inert |
| Rate saturation auxiliary | maximum 0; stored half-value 1 AU | Inert | Inert |

The configured built-in decoder parameter (`decoder_gain_raw`, effective gain
8) is part of checkpoint state, but no reported training or assay result uses
`net.decode()`. The task loss computes its own gain-free population-vector
alignment directly from noisy L2/3 rates; the assay fits a separate
condition-blind cosine-nearest-centroid decoder. This distinction prevents the
stored gain from being mistaken for a trained decoding advantage.

Each arm loads the exact same seed-specific common state and receives a fresh
Adam optimizer. No checkpoint is selected from validation behavior; the fixed
final step is used.

## Training objective

Let integer `channels` have shape `[B,S]`, logits have shape `[B,S,36]`, and
raw L2/3 rates `r` have shape `[B,S,36]` in arbitrary activity units (AU).

Next-step prediction consumes only `t=0,...,S-2`:

`Lpred = mean CE(logits[:, :-1, :], channels[:, 1:])`.

For current-orientation precision, one independent noise tensor
`eta ~ Normal(0, sigma_train^2)` is added before rectification. With
`a=ReLU(r+eta)`, orientation phases `phi_i=2 pi i/36`, and target phase
`phi_y`, define

`z = sum_i a_i exp(j phi_i)`,

`c = Re(z exp(-j phi_y)) / (|z| + 1e-8 * 36 * R_ref)`,

`Lpv = mean(1 - c)`.

The task and rate terms are

`T = 0.5 Lpred/log(36) + 0.5 Lpv/2`,

`E = mean(r)/R_ref`,

`J(alpha) = (1-alpha) T + alpha E`.

`R_ref` is the mean initialized, no-feedback L2/3 activity over all 36
orientations and channels. `A_ref` is the median orientation-wise maximum on
that grid, and `sigma_train=0.25 A_ref`. The population-vector term uses all
sequence time steps. `E` averages over batch, time, and all 36 channels,
including `t=0`; because the prior feedback state is zero, that first rate has
no feedback-dependent gradient. `E` is a **normalized L2/3 mean-rate proxy**.
It is dimensionless after normalization, but it is not ATP use, oxygen
consumption, synaptic energy, or whole-network metabolic cost.

### Training sequence process and random streams

Training sees only the ordinary momentum generator, never assay A/B labels or
histories. For each sequence:

1. `a[0]` is uniform on `{-1,0,+1}` channel-step acceleration.
2. At each later index, the previous acceleration index is retained with
   probability `0.9`; otherwise a new value is sampled uniformly from the same
   three-value set. Because replacement can redraw the old value, the total
   probability of an unchanged value is `0.9 + 0.1/3`.
3. `v[0]` is uniform on the integer channel steps `{-4,...,+4}`. Then
   `v[t]=clip(v[t-1]+a[t-1],-4,+4)`.
4. The initial orientation channel is uniform on `{0,...,35}` and
   `y[t]=(y[0]+sum_{k<t} v[k]) mod 36`. The model receives `5*y[t]` nominal
   degrees. The executed batch size is 128 and sequence length is 12.

For experimental seed `s`, global PyTorch CPU and CUDA seeds are set to `s`.
Device-local generators use `200000+s` (pretrain sequences), `300000+s`
(pretrain task noise), `400000+s` (every alpha arm's sequences), and
`500000+s` (every alpha arm's task noise). Reinitializing every alpha arm with
the same two latter seeds gives common random numbers across alpha values;
each arm still has its own generator object and optimizer. Descriptive feedback
statistics use `800000+s` and do not affect optimization. The assay decoder
uses fixed device-local noise seeds `910001` for fitting and `910002` for
testing, with one shared A/B noise table within each split.

PyTorch deterministic algorithms and deterministic cuDNN behavior are
requested. Generator streams are device-specific, so CPU and CUDA runs are not
claimed to be numerically identical. The reported run kept the model, global
orientation tensors, generators, and inputs on `cuda:0`. A CPU-only
installation auto-selects CPU; on a CUDA-visible host, use the reported
`cuda:0` path rather than forcing `--device cpu`, because imported global
orientation tensors are created on the auto-selected device before CLI device
selection.

## Historical six-alpha development protocol

- Exploratory development seeds: `0,1,2,3`.
- Device used for the reported run: `cuda:0` on an NVIDIA GeForce RTX 5090.
- Common pretrain: 3,000 steps, batch 128, sequence length 12.
- Each alpha arm: 8,000 steps, batch 128, sequence length 12.
- Adam learning rate: `1e-3`. Gradient clip `5.0` is the declared trainer and
  reproduction value; the saved source summaries do not independently record
  or validate that clip setting.
- Feedback mode: `posterior_prior_excess` in every arm.
- Local competition: frozen and byte-checked against the common pretrain.
- Deterministic PyTorch algorithms and deterministic cuDNN settings are
  requested. Cross-version, driver, or CPU-versus-CUDA bit identity is not
  claimed.

### Current and legacy CLI defaults

The trainer CLI makes the historical six-alpha,
`posterior_prior_excess`, frozen-local-competition protocol the trainer CLI
default. Thus an invocation that omits those three options resolves to
`--alphas 0.0 0.1 0.3 0.5 0.7 0.9`,
`--feedback-mode posterior_prior_excess`, and `--freeze-local-comp`.

The numerical kernels and losses are unchanged for explicit reported
commands. To request the legacy five-arm behavior explicitly, use:

```bash
python tools/train_emergent_task_energy_axis.py \
  --feedback-mode baseline \
  --no-freeze-local-comp \
  --alphas 0.1 0.3 0.5 0.7 0.9
```

That command selects a different experimental protocol; it is not the source
of the reported posterior-excess/frozen-competition results in this document.

## Operational continuation/reversal assay

For every final channel `y` and
`v in {-3,-2,-1,+1,+2,+3}`, the fixed length-five histories are

`A = [y-4v, y-3v, y-2v, y-v, y] mod 36`,

`B = [y+2v, y+v, y, y-v, y] mod 36`.

The 36 final channels crossed with six velocities give 216 matched pairs. `A`
is an operational constant-velocity **continuation**. `B` is an operational
**reversal**: its last velocity change is `2v`, outside the training
acceleration support. It is therefore an operational out-of-distribution (OOD)
reversal, not a sample from the ordinary training generator.

The scientific names used here are **operational continuation A** and
**matched operational OOD reversal B**. Some retained JSON keys use the legacy
short names `expected_A` and `unexpected_B`; those are schema labels only. This
workflow does not gate pairs on predictor probability, so it does not establish
that the model assigns a higher probability to each A history, nor that A/B
span biological expected/unexpected stimuli in general.

### Three readout families

1. **Mean-rate proxy.** Define the activity-unit guard
   `epsilon_rate=1e-8*N*R_ref`. Within one seed and arm, first compute the two
   condition means over all 216 final responses and all 36 L2/3 channels,
   `mu_A=mean_{p,i}(r_A[p,i])` and `mu_B=mean_{p,i}(r_B[p,i])`. Stored saving is
   the paired ratio of means
   `saving=(mu_B-mu_A)/(mu_B+epsilon_rate)`. The phase-space y coordinate uses the
   identical denominator and opposite numerator,
   `y=(mu_A-mu_B)/(mu_B+epsilon_rate)=-saving`. This is not the mean of 216
   pair-specific ratios. Across seeds, the plot reports the mean and sample SEM
   of the four independently computed seed ratios.
2. **Condition-blind, noise-held-out orientation decoding.** For each split,
   noisy trial features are
   `x=ReLU(r+eta)` with `eta~Normal(0,sigma_train^2)`, where `sigma_train` is
   loaded from the checkpoint's activity references. Features then receive
   per-trial L1 normalization followed by L2 normalization. The train split
   uses 32 repeats and fixed seed
   `910001`; the test split uses 32 independent repeats and fixed seed
   `910002`. Within a split, A and B receive the same noise table, preserving
   their pairing. One balanced cosine-nearest-centroid readout is fit to pooled
   A+B training features, and the separately noised A and B test features are
   scored against that one shared readout. Only additive noise is held out: the
   same 216 histories underlie fitting and testing, so this is not
   stimulus-held-out, orientation-held-out, or history-held-out decoding.
   Chance is `1/36`.
3. **Aligned profile shape.** Final rates are circularly aligned to `y`. For raw
   profiles, `C` is the mean at offsets `{-1,0,+1}` channels and `F` is the mean
   at `{-6,-5,-4,-3,+3,+4,+5,+6}`. With 5 degrees/channel these are
   `C={-5,0,+5}` degrees and `F={±15,±20,±25,±30}` degrees. For each aligned
   profile, define `q_i=r_i/(sum_j r_j+epsilon_rate)`. `Cq` and `Fq` are the means
   of `q` in the same center and flank windows, and
   `Q=(Cq-Fq)/(Cq+Fq+1e-8)`.

The seed is the independent unit. Plots first average 216 rows within each seed
and then show the four-seed mean with sample SEM.

## Literal first-stimulus tuning baseline

The gray tuning comparator is the **first stimulus (no prior context; normal
feedback-on unroll, feedback state=0)**, not reversal B and not a feedback-off
final response. It is the same trained endpoint's ordinary response at `t=0`,
where hidden, feedback, and adaptation states are zero. For seed `s`, align each
A response to `A[0]`, each B response to `B[0]`, average within history type,
and pool symmetrically:

`baseline_s = 0.5 * (mean align(r_A,t=0, A[0]) + mean align(r_B,t=0, B[0]))`.

The independently recomputed baseline is bit-identical for all four seeds and
both displayed endpoints, as expected from the fixed feedforward substrate and
frozen local competition. The plotter asserts that equality instead of
hardcoding or copying a curve. An isolated one-frame orientation sweep is an
equivalent invariant check, not the primary definition.

The x axis is the **nominal fixed feedforward orientation preference relative
to presented orientation**. These are aligned population response profiles,
not longitudinal single-neuron tuning-curve measurements.

## Dampening calibration and fresh confirmation

### What was and was not changed

The architecture, training data, optimizer, task term, and rate term are the
same in every arm. The only difference is the scalar `alpha` in

`J(alpha) = (1 - alpha) T + alpha E`.

Thus `alpha=0.5` is the balanced normalized task/mean-rate coordinate, not a
third loss, a response fit, or an expected-stimulus-specific intervention. The
assay labels, literal `t=0` baseline, center/flank windows, and acceptance
thresholds are absent from training.

The scientific decision uses three readout families:

1. **Energy:** `rate_A < rate_B`.
2. **Decoding:** one pooled condition-blind decoder has `decode_A < decode_B`,
   with both accuracies above `1/36` chance.
3. **Shape:** with `Cret=C_A/C_t0`, `Fret=F_A/F_t0`, and assay contrasts
   `dC=C_A-C_B`, `dF=F_A-F_B`, `dQ=Q_A-Q_B` in their documented
   normalizations, require `Cret<Fret`, `Cret<1`, `dC<dF`, and `dQ<0`.

The shape family also guards against accepting near-total activity collapse.
Define whole-profile retention

`M = AUC(A final aligned 36-bin profile) / AUC(t0 aligned 36-bin profile)`

which is exactly `rate_A/rate_t0` because the bins have equal width. Relative
to the more energy-dominated `alpha=0.9` comparator, each `alpha=0.5` seed must
have `M` ratio at least `1.25`, `M` difference at least `.040`, `Fret` ratio at
least `1.15`, and `Fret` difference at least `.040`; the cohort mean `M` must be
at least `.250`. These are post-training amplitude safeguards within the shape
validation, not extra optimization terms.

### Calibration lineage

| Stage | Seeds | Coordinate | Outcome |
| --- | --- | ---: | --- |
| Development calibration | `0–3` | `.6` | Development screen passed; mean `M=.2740319260531955` |
| First fresh cohort | `4–7` | `.6` vs `.9` | Stored energy, decoding, `dC<dF`, and `dQ<0` checks passed; amplitude retention failed, so `.6` was rejected |
| Revised development | `4–7` | `.5` vs `.9` | All three families, including amplitude safeguards, passed |
| Independent from-scratch confirmation | `8–11` | `.5` vs `.9` | Every per-seed criterion and the cohort criterion passed |

For the fresh `.6` cohort, mean `M(.6)=.21336743856557555` and mean
`M(.9)=.15607987727316153`. Seed 5 failed the `1.25` M-ratio boundary at
`1.2243398680161324`; seed 7 failed it at `1.206259035990063` and also failed
the `.040` M-difference boundary at `.029167501148376213`; cohort mean `M(.6)`
was below `.250`. The read-only `.6` evaluation seal is
`982248457917af129728694e732cfb83412c8ae3d54e770666b826a07afdd6ae`;
the development assay-manifest hash is
`9cd24742493d4ecab400a16149e52334a8e8d53d4a4a656497b8a40962358957`.
Because `Cret/Fret` was not a stored verified leaf in that `.6` evaluation, no
claim about those two quantities is made here.

The revised-development assay seal is
`81440ae260df5bebb39736417683b1b3803f0a1b7f73acba6b82532f29c301fc`
and its training seal is
`10c891e303f98d1d81459f496434b75fe6728e9f72a2a91db8c2afbfde09f57c`.
The fresh cohort was not reused from either development stage.

### Exact fresh values

The following are the frozen corrected-evaluator values for seeds `8–11`.
`rate` is final mean L2/3 activity in AU; decoding is top-1 accuracy.

| Seed | rate A | rate B | B−A | decode A | decode B | B−A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | .04836434524352033 | .08832483409471346 | .03996048885119313 | .2074652761220932 | .30975115299224854 | .10228587687015533 |
| 9 | .04837169788107353 | .08525280885222897 | .036881110971155436 | .25231480598449707 | .30787035822868347 | .0555555522441864 |
| 10 | .04537613037312018 | .08332933417317500 | .03795320380005481 | .20717592537403107 | .27994790673255920 | .07277198135852814 |
| 11 | .05732431760151368 | .09762491445172103 | .04030059685020735 | .22714120149612427 | .32262730598449707 | .09548610448837280 |

| Seed | Cret | Fret | Fret−Cret | dC | dF | dF−dC | dQ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | .11900846191199754 | .49225945140670060 | .37325098949470303 | -1.9968568480174846 | -.04561513467238416 | 1.9512417133451005 | -.8198121729325180 |
| 9 | .11548944034021766 | .49252814065387730 | .3770387003136596 | -1.7802617163520735 | -.07816464260593912 | 1.7020970737461345 | -.7093986943501684 |
| 10 | .11391838910341391 | .45549921932916126 | .34158083022574737 | -1.6127748803014170 | -.13607883114835465 | 1.4766960491530623 | -.6465777749851730 |
| 11 | .14345936502608167 | .57908956790514540 | .43563020287906373 | -2.2764149532311470 | +.06769186699907595 | 2.3441068202302230 | -.8795613449976969 |

| Seed | M(.5) | M(.9) | M ratio | M difference | Fret(.5) | Fret(.9) | Fret ratio | Fret difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | .2901656344312839 | .1948357860747792 | 1.4892830535757764 | .09532984835650471 | .49225945140670060 | .30363745346831800 | 1.6212079431698425 | .18862199793838258 |
| 9 | .29020974714964337 | .12392075590220045 | 2.3418978123300020 | .16628899124744292 | .49252814065387730 | .22144169945569578 | 2.2241887678089203 | .27108644119818150 |
| 10 | .2722376079208284 | .1332041298063895 | 2.0437625193492300 | .13903347811443892 | .45549921932916126 | .23864778729485928 | 1.9086672643914901 | .21685143203430200 |
| 11 | .34392168241773374 | .17233959879560087 | 1.9956045205004427 | .17158208362213287 | .57908956790514540 | .27084379648146545 | 2.1380942647684900 | .30824577142368000 |

The weakest observed margins were still positive:

| Criterion | Weakest margin above the passing boundary |
| --- | ---: |
| Energy `B−A > 0` | .036881110971155436 |
| Decoding `B−A > 0` | .0555555522441864 |
| Decode A above chance | .17939814759625328 |
| Decode B above chance | .25217012895478140 |
| `Fret−Cret > 0` | .34158083022574737 |
| `dF−dC > 0` | 1.4766960491530623 |
| `-dQ > 0` | .6465777749851730 |
| `1−Cret > 0` | .8565406349739183 |
| M ratio above `1.25` | .2392830535757764 |
| M difference above `.040` | .05532984835650471 |
| Fret ratio above `1.15` | .4712079431698426 |
| Fret difference above `.040` | .14862199793838257 |
| Cohort mean M above `.250` | .04913366797987234 |

The four `M(.5)` values average to `0.29913366797987234`. All `48/48`
per-seed checks and the cohort check passed.

### Evaluator correction and sealed evidence

The first frozen evaluator incorrectly bound the symbol `M` to the already
stored A/B saving ratio. That was an evaluator-definition defect: the model,
checkpoints, 216-pair assays, profiles, thresholds, and training were unchanged.
The corrected evaluator restores the development definition shown above,
`AUC(A final)/AUC(t0)`, equivalently `rate_A/rate_t0`. The original evaluator,
result, and log are preserved; a hash-bound supersession records the correction.

| Evidence | SHA-256 |
| --- | --- |
| Final all-assays seal | `027feb665537e1f54628e9e7af1ff5b25bdb759e067ff02e6b751fb42e37cd51` |
| 32-entry assay ledger | `04404bd8efdaba8a506b686d746c79bbb03b4212799ced43fd3c8ef2c3fb77a4` |
| Corrected frozen result | `8bd3c6dd13cd3770e86eed23ef7e0b1d8103990fcc31b6162cf6a043611b2d7d` |
| Evaluator supersession | `c3fa958cba809e0aafef0c2a8db6de4224c05b8610ddf5320ac01feaba0284ee` |
| M-definition regression result | `07a5874bce40819ef64eaf61055385307762b59d1df3ac0dac980222d480182e` |
| Training seal | `2de0c984c0346f39e4bf82aebad814bc76b03057b00ecf432819164669ea557b` |
| 58-entry training ledger | `f248a263ea285cce5e0ad16db2fb95a357cee2c1705a35f66b9f6e6eae53b32b` |

Ledger verification was `32/32`; an independent read-only RTX 5090 replay
matched all `376/376` official numeric leaves exactly and reproduced all frozen
gate decisions. These hashes identify an external result bundle; no
developer-specific absolute path is part of the scientific claim.

## Historical six-alpha development sweep

This retained exploratory sweep is not the fresh confirmation above. Values
below are mean ± sample SEM over development seeds `0,1,2,3`. `Delta` means
continuation A minus reversal B, except saving, which is
`(B-A)/(B+epsilon_rate)`. `DeltaC`
and `DeltaF` are normalized by `R_ref`. The table is rendered from the portable
`figures/emergent_reference_comparison/all_alpha_assay_summary.json`, which is
itself derived from the four per-seed `endpoint_assay.json` audit records in
provenance branch A above.

| alpha | Decode A | Decode B | Delta decode | Saving | Delta C | Delta F | Delta Q |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | .9975 ± .0005 | .3439 ± .0133 | +.6535 ± .0136 | .0761 ± .0011 | +1.8704 ± .0373 | -.7383 ± .0121 | +.2737 ± .0042 |
| 0.1 | .9849 ± .0018 | .4579 ± .0050 | +.5270 ± .0035 | .0346 ± .0003 | -1.6403 ± .0379 | +.2166 ± .0112 | -.2368 ± .0094 |
| 0.3 | .4987 ± .0096 | .2253 ± .0101 | +.2734 ± .0181 | .3220 ± .0041 | -3.2448 ± .0634 | +.5189 ± .0186 | -1.1700 ± .0249 |
| 0.5 | .2583 ± .0147 | .3024 ± .0067 | -.0442 ± .0111 | .4210 ± .0122 | -1.8045 ± .1846 | -.0591 ± .0243 | -.6883 ± .0596 |
| 0.7 | .1800 ± .0250 | .2823 ± .0089 | -.1023 ± .0174 | .4713 ± .0535 | -1.1135 ± .1663 | -.2254 ± .0227 | -.4969 ± .0997 |
| 0.9 | .1469 ± .0122 | .2459 ± .0069 | -.0990 ± .0144 | .5422 ± .0200 | -.8912 ± .0661 | -.2780 ± .0158 | -.4232 ± .0491 |

This is not a monotone two-state interpolation. `alpha=0.1` and `0.3` retain a
positive A-minus-B decoding contrast while already having negative `DeltaC`
and `DeltaQ`; they are mixed arms. The decoding contrast crosses between `0.3`
and `0.5`. Saving is also nonmonotone at `0.1` before rising from `0.3` onward.

### Historical alpha=.0 versus alpha=.9 curves relative to t0

This subsection belongs only to the retained seeds `0–3` historical replay;
the current figures use `.0` and selected `.5`. Ratios here are computed from
the historical across-seed mean curves. Delta uncertainties are seed SEM.

| Curve | C (AU) | F (AU) | C/F | Delta C vs t0 | Delta F vs t0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared first stimulus, zero prior feedback/adaptation | .779353 | .279427 | 2.789114 | — | — |
| Task-only endpoint (α=0.0) | 1.059204 | .266956 | 3.967706 | +.279851 ± .005268 | -.012470 ± .000265 |
| Rate-cost-weighted endpoint (α=0.9) | .046336 | .072884 | .635756 | -.733017 ± .002635 | -.206543 ± .002781 |

The task-only endpoint has a higher center, slightly lower flanks, and a larger
C/F ratio: a sharpening-like measured pattern. The rate-cost-weighted endpoint
retains only about 5.9% of the t0 center and 26.1% of the t0 flanks. Its center
is preferentially suppressed and C/F drops below one, but its flanks are not
preserved in absolute units. The precise description is **broad attenuation
with preferential center suppression**, not absolute flank-preserving
dampening.

The phase-space x coordinate is `Δ decode accuracy (continuation A − reversal
B)`. Its y coordinate is `Δ final mean L2/3 rate ((continuation A − reversal
B)/(reversal B + epsilon_rate))`. The endpoint values are:

- task-only endpoint (α=0.0): `Delta decode=.653537 ± .013624`,
  `(A-B)/(B+epsilon_rate)=-.076096 ± .001145`;
- rate-cost-weighted endpoint (α=0.9): `Delta decode=-.098995 ± .014371`,
  `(A-B)/(B+epsilon_rate)=-.542170 ± .019979`.

## Biology and engineering mapping

| Model element | Permitted interpretation |
| --- | --- |
| Fixed 36-channel L4/L2/3 basis | Engineering abstraction of circular orientation preference; not measured receptive fields or cortical magnification |
| L2/3 nonnegative rate | Population activity variable in AU; not a spike train, membrane voltage, or identified neuron |
| GRU and `W_fb` | Engineering temporal predictor; not an anatomical higher visual area or a specified feedback tract |
| Five-gain motif | SOM/VIP-inspired **Dale-sign-constrained rate motif**; sign analogy only, not a fitted cell-type circuit |
| Divisive local competition | Engineering normalization mechanism with a cortical-normalization analogy |
| `mean(r)/R_ref` | **Normalized L2/3 mean-rate proxy**; not a biophysical energy budget |
| Continuation/reversal | Operational history construction; reversal is OOD relative to training acceleration support |

Primary literature motivates qualitative hypotheses and sign analogies only.
It does not fit the constants, validate the software as biology, establish
identified cell types, or turn operational A/B into subjective expectation:

- VIP-to-SOM disinhibitory connectivity: Pi et al. (2013),
  [doi:10.1038/nature12676](https://doi.org/10.1038/nature12676), and Pfeffer et
  al. (2013), [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446).
- Cortical response normalization: Heeger (1992),
  [doi:10.1017/S0952523800009640](https://doi.org/10.1017/S0952523800009640),
  and Carandini, Heeger & Movshon (1997),
  [doi:10.1523/JNEUROSCI.17-21-08621.1997](https://doi.org/10.1523/JNEUROSCI.17-21-08621.1997).
- Expectation-related sharpening and suppression motivate the comparison, not
  a fitted target: Kok et al. (2012),
  [doi:10.1016/j.neuron.2012.04.034](https://doi.org/10.1016/j.neuron.2012.04.034),
  and Alink et al. (2010),
  [doi:10.1523/JNEUROSCI.3730-10.2010](https://doi.org/10.1523/JNEUROSCI.3730-10.2010).
- Neural signaling has empirically measured energy–information trade-offs:
  Laughlin, de Ruyter van Steveninck & Anderson (1998), *The metabolic cost of
  neural information*, [PMID 10195106](https://pubmed.ncbi.nlm.nih.gov/10195106/),
  and Niven, Anderson & Laughlin (2007), *Fly photoreceptors demonstrate
  energy-information trade-offs in neural coding*,
  [PMID 17373859](https://pubmed.ncbi.nlm.nih.gov/17373859/).
- Activity also imposes local ATP demand at synapses: Rangaraju, Calloway & Ryan
  (2014), *Activity-driven local ATP synthesis is required for synaptic
  function*, [PMID 24529383](https://pubmed.ncbi.nlm.nih.gov/24529383/).
  These studies motivate an energetic constraint; they do **not** validate
  `mean(r)/R_ref` as ATP consumption. That term remains an engineering proxy.

## RTX 5090 reproduction recipes

### Fresh alpha=0.5 confirmation

Run from the repository root. Keep generated checkpoints outside the checkout.
The command below reproduces the fresh two-arm training and assay protocol with
portable operator-selected paths; it does not reuse development checkpoints.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
CONFIRM_ROOT="${CONFIRM_ROOT:-$HOME/neuroips_runs/task_energy_alpha_0p5_confirmation}"
case "$CONFIRM_ROOT" in
  "$PWD"|"$PWD"/*)
    echo "CONFIRM_ROOT must be outside the repository: $CONFIRM_ROOT" >&2
    exit 1
    ;;
esac
if [ -e "$CONFIRM_ROOT" ]; then
  echo "Refusing to reuse existing CONFIRM_ROOT: $CONFIRM_ROOT" >&2
  exit 1
fi
mkdir -p "$CONFIRM_ROOT"

for SEED in 8 9 10 11; do
  python -B tools/train_emergent_task_energy_axis.py \
    --seed "$SEED" \
    --device cuda:0 \
    --out "$CONFIRM_ROOT" \
    --pretrain-steps 3000 \
    --axis-steps 8000 \
    --batch 128 \
    --sequence-length 12 \
    --lr 0.001 \
    --clip 5.0 \
    --log-every 100 \
    --checkpoint-every 250 \
    --alphas 0.5 0.9 \
    --freeze-local-comp \
    --feedback-mode posterior_prior_excess
done

for SEED in 8 9 10 11; do
  python -B tools/assay_emergent_task_energy_axis.py \
    --run-dir "$CONFIRM_ROOT/seed_$SEED" \
    --device cuda:0 \
    --out "$CONFIRM_ROOT/seed_$SEED/endpoint_assay.json" \
    --alphas 0.5 0.9
done
```

Each seed's two arms load the same common pretrain and use common random-number
streams, but have separate fresh optimizers. Confirm that the trainer process
has a live PID and its log has emitted output before describing a run as
started; command submission alone is not evidence that GPU work is active.

Run the validated portable gate evaluator after the assays:

```bash
python -B tools/evaluate_emergent_task_energy_gates.py \
  --run-dir "$CONFIRM_ROOT/seed_8" \
  --run-dir "$CONFIRM_ROOT/seed_9" \
  --run-dir "$CONFIRM_ROOT/seed_10" \
  --run-dir "$CONFIRM_ROOT/seed_11" \
  --candidate-alpha 0.5 \
  --comparator-alpha 0.9 \
  --device cuda:0 \
  --out "$CONFIRM_ROOT/frozen_gate_decision.json"
```

The evaluator's 10 focused tests passed. An independent sealed CUDA replay on
seeds `8–11` passed and matched the authoritative result exactly: `104` metric,
`16` comparator, `48` gate, `4` seed-status, and `2` cohort leaves, plus `44`
seal-binding leaves. See the
[focused evaluator tests](../tests/test_evaluate_emergent_task_energy_gates.py)
and [selection-record schema and binding
tests](../tests/test_endpoint_selection_record_schema.py). The output is a
portable gate-decision JSON without host paths. It reproduces the scientific
gate decision; it does not regenerate the historical external assay ledger or
final seal, which remain bound by the sealed evidence above.

### Matched alpha=0.0 comparator and reference-layout plots

The sharpening comparator must use each confirmation seed's byte-identical
common pretrain. Build a separate plotting bundle so the sealed confirmation
directory remains read-only. This subsection defines all of its inputs; both
default output locations are outside the checkout, so a reproduction does not
overwrite tracked figures:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
CONFIRM_ROOT="${CONFIRM_ROOT:-$HOME/neuroips_runs/task_energy_alpha_0p5_confirmation}"
PLOT_ROOT="${PLOT_ROOT:-$HOME/neuroips_runs/task_energy_alpha_0p5_plot_bundle}"
FIGURE_OUT="${FIGURE_OUT:-$HOME/neuroips_outputs/task_energy_reference_figures}"
if [ ! -d "$CONFIRM_ROOT" ]; then
  echo "Missing confirmation directory: $CONFIRM_ROOT" >&2
  exit 1
fi
case "$PLOT_ROOT" in
  "$PWD"|"$PWD"/*)
    echo "PLOT_ROOT must be outside the repository: $PLOT_ROOT" >&2
    exit 1
    ;;
esac
if [ -e "$PLOT_ROOT" ]; then
  echo "Refusing to reuse existing PLOT_ROOT: $PLOT_ROOT" >&2
  exit 1
fi

for SEED in 8 9 10 11; do
  mkdir -p "$PLOT_ROOT/seed_$SEED"
  cp "$CONFIRM_ROOT/seed_$SEED/common_pretrain_final.pt" \
     "$PLOT_ROOT/seed_$SEED/common_pretrain_final.pt"
  cp "$CONFIRM_ROOT/seed_$SEED/alpha_0p5_final.pt" \
     "$PLOT_ROOT/seed_$SEED/alpha_0p5_final.pt"

  python -B tools/train_emergent_task_energy_axis.py \
    --seed "$SEED" \
    --device cuda:0 \
    --out "$PLOT_ROOT" \
    --pretrain-steps 3000 \
    --axis-steps 8000 \
    --batch 128 \
    --sequence-length 12 \
    --lr 0.001 \
    --clip 5.0 \
    --log-every 100 \
    --checkpoint-every 250 \
    --alphas 0.0 \
    --freeze-local-comp \
    --feedback-mode posterior_prior_excess
done

python -B tools/plot_emergent_reference_figures.py \
  --run-dir "$PLOT_ROOT/seed_8" \
  --run-dir "$PLOT_ROOT/seed_9" \
  --run-dir "$PLOT_ROOT/seed_10" \
  --run-dir "$PLOT_ROOT/seed_11" \
  --task-alpha 0.0 \
  --energy-alpha 0.5 \
  --device cuda:0 \
  --out-dir "$FIGURE_OUT"
```

The plotter obtains the gray curve by replaying the literal `t=0` response; it
does not substitute reversal B or a feedback-off final response. Expected
outputs are `plot_data.json`, `tuning_dampening.png`,
`tuning_sharpening.png`, `1_decode_signflip.png`, and
`3_decode_energy_phasespace.png`.

### Historical six-alpha sweep

Run from the repository root with a fresh output directory. The guard prevents
accidentally resuming or mixing with an existing run.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
RUN_ROOT="${RUN_ROOT:-$HOME/neuroips_runs/emergent_task_energy_axis_rtx5090_reproduction}"
case "$RUN_ROOT" in
  "$PWD"|"$PWD"/*)
    echo "RUN_ROOT must be outside the repository: $RUN_ROOT" >&2
    exit 1
    ;;
esac
if [ -e "$RUN_ROOT" ]; then
  echo "Refusing to reuse existing RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

for SEED in 0 1 2 3; do
  python tools/train_emergent_task_energy_axis.py \
    --seed "$SEED" \
    --device cuda:0 \
    --out "$RUN_ROOT" \
    --pretrain-steps 3000 \
    --axis-steps 8000 \
    --batch 128 \
    --sequence-length 12 \
    --lr 0.001 \
    --clip 5.0 \
    --log-every 100 \
    --checkpoint-every 250 \
    --alphas 0.0 0.1 0.3 0.5 0.7 0.9 \
    --freeze-local-comp \
    --feedback-mode posterior_prior_excess
done

for SEED in 0 1 2 3; do
  python tools/assay_emergent_task_energy_axis.py \
    --run-dir "$RUN_ROOT/seed_$SEED" \
    --device cuda:0 \
    --out "$RUN_ROOT/seed_$SEED/endpoint_assay.json" \
    --alphas 0.0 0.1 0.3 0.5 0.7 0.9
done

: "${GENERATED_AT:?Set GENERATED_AT to an explicit ISO-8601 timestamp}"
python tools/aggregate_emergent_task_energy_assays.py \
  --run-dir "$RUN_ROOT/seed_0" \
  --run-dir "$RUN_ROOT/seed_1" \
  --run-dir "$RUN_ROOT/seed_2" \
  --run-dir "$RUN_ROOT/seed_3" \
  --generated-at "$GENERATED_AT" \
  --out "$RUN_ROOT/figures/emergent_reference_comparison/all_alpha_assay_summary.json"

python tools/plot_emergent_reference_figures.py \
  --run-dir "$RUN_ROOT/seed_0" \
  --run-dir "$RUN_ROOT/seed_1" \
  --run-dir "$RUN_ROOT/seed_2" \
  --run-dir "$RUN_ROOT/seed_3" \
  --task-alpha 0.0 \
  --energy-alpha 0.9 \
  --device cuda:0 \
  --out-dir "$RUN_ROOT/figures/emergent_reference_comparison"
```

The repository intentionally has no blanket checkpoint/output ignore rule, so
the command requires `RUN_ROOT` outside the checkout. Expected per-seed
artifacts include `common_pretrain_final.pt`, six
`alpha_*_final.pt` checkpoints, `training.jsonl`, `training_summary.json`, and
`endpoint_assay.json`. The compact all-alpha assay summary is produced by
`tools/aggregate_emergent_task_energy_assays.py` from the four assay JSON audit
records, with training/checkpoint files read only for protocol and hash
provenance, at
`figures/emergent_reference_comparison/all_alpha_assay_summary.json`.
This historical `alpha=0.0` versus `0.9` replay produces `plot_data.json` and:

- `tuning_dampening.png` (historical requested filename; content is the honest
  high-alpha broad-attenuation profile);
- `tuning_sharpening.png`;
- `1_decode_signflip.png`;
- `3_decode_energy_phasespace.png`.

Reproduced checkpoints and logs are untracked artifacts below the fresh
`RUN_ROOT` selected by the operator; no developer-specific absolute checkpoint
path is canonical. The tracked aggregate presentation lives under
`figures/emergent_reference_comparison/` in this repository.

## Limitations

- Four development seeds plus four independent fresh seeds establish a
  computational result for this implementation, not confirmatory biological
  evidence.
- Reversal B is OOD relative to the training acceleration support, while A is
  an in-support constant-velocity history. This asymmetry and the absence of a
  per-pair predictor-probability gate prevent general expected/unexpected-stimulus
  claims.
- Condition-blind decoding generalizes across independent noise tables only;
  the histories, orientations, and labels are reused in fitting and testing.
- The mean-rate objective is not a physical energy model.
- The motif is a sign-constrained rate analogy; no interneuron identity,
  synaptic conductance, spike timing, laminar anatomy, or causal biology is
  inferred.
- GRU, `W_fb`, and motif gains co-adapt. A gain-only explanation is invalid.
- The alpha series is not monotonic in every metric, and intermediate arms are
  mixed regimes.
- At `alpha=0.5`, both center and flank retention are below one. The supported
  statement is center suppression with **relative** flank sparing, not absolute
  flank preservation or enhancement above baseline.
- The plotted endpoints are selected coordinates (`alpha=0.0` and `0.5`), not
  evidence of a sharp phase boundary. The historical six-alpha series is
  descriptive and nonmonotonic in some metrics.
- The plotter checks checkpoint schema and recomputes selected metrics, but the
  scientific acceptance decision comes from the separate fresh-cohort assay
  evaluation, not visual resemblance or `plot_data.json`.
- Standalone assay records do not store raw aligned profiles. Consequently the
  all-alpha artifact contains their scalar metrics, not reconstructed profiles;
  36-bin profiles in `plot_data.json` cover only the displayed endpoints.
- Exact reproducibility depends on compatible PyTorch, CUDA, driver, and GPU
  behavior; the repository does not pin a complete binary environment.
