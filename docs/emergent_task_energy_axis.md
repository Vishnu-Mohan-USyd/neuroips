# Current task–energy axis workflow

This is the canonical guide to the repository's current four-seed workflow.
The older Phase A/Phase B and repair notes remain useful historical lineage, but
they do not define the model, checkpoints, assays, or figures described here.

## Newcomer overview

The experiment asks what one recurrent orientation circuit learns when the same
task objective is traded against a **normalized L2/3 mean-rate proxy**. Every
optimization arm starts from the same seed-specific task pretrain and uses the
same architecture. The only arm coordinate is

`J(alpha) = (1 - alpha) T + alpha E`,

for `alpha in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}`. Training never receives the
post-hoc continuation/reversal labels, matched assay pairs, endpoint names, or
target response shapes.

Across four development seeds, the **task-only endpoint (α=0.0)** decodes
operational continuation A better than matched operational OOD reversal B and
has a sharpening-like aligned profile. The **rate-cost-weighted endpoint
(α=0.9; 10% task, 90% normalized L2/3 mean-rate proxy)** has a lower A/B
mean-rate ratio and decodes A less accurately than B. Its absolute profile is
broad attenuation with preferential center suppression; absolute flanks are
not preserved relative to the first-stimulus baseline. Intermediate arms are
mixed rather than two clean categories. These are post-training measurements,
not labels supplied to the loss.

## Artifact provenance

Executable code and checkpoint contents define the computation. The measured
outputs then follow two distinct provenance branches:

```text
(A) trained checkpoints
    └─ tools/assay_emergent_task_energy_axis.py replay
       └─ per-seed endpoint_assay.json audit records
          └─ tools/aggregate_emergent_task_energy_assays.py
             └─ figures/emergent_reference_comparison/all_alpha_assay_summary.json
                └─ canonical six-alpha table in this guide

(B) selected alpha=0.0 and alpha=0.9 checkpoints
    └─ tools/plot_emergent_reference_figures.py direct replay
       └─ in-memory aggregates
          ├─ figures/emergent_reference_comparison/plot_data.json
          └─ four sibling PNG figures
```

Per-seed `training_summary.json` files record configuration, state hashes, and
training diagnostics. They are not measurement or plotter inputs; the compact
aggregator reads them only to corroborate protocol and checkpoint provenance.
Its numerical measurements come from `endpoint_assay.json`. The plotter
directly remeasures the two displayed checkpoints, writes `plot_data.json` and
the PNGs as sibling outputs, and does not apply a phenotype acceptance gate.
Visual resemblance is therefore not itself a validation decision.

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
| [`tools/tuned_emergence_lib.py`](../tools/tuned_emergence_lib.py) | Fixed orientation basis, L2/3 rate circuit, recurrent predictor, feedback transform, and causal unroll timing |
| [`tools/train_emergent_task_energy_axis.py`](../tools/train_emergent_task_energy_axis.py) | Common task pretrain, six task–energy arms, optimizer policy, deterministic streams, and checkpoints |
| [`tools/assay_emergent_task_energy_axis.py`](../tools/assay_emergent_task_energy_axis.py) | Fixed 216-pair operational continuation/reversal assay and its three readout families |
| [`tools/aggregate_emergent_task_energy_assays.py`](../tools/aggregate_emergent_task_energy_assays.py) | Schema 1.0.0 portable summary of existing per-seed assay fields and aggregate mean/sample SEM, with logical artifact IDs and hashes |
| [`tools/plot_emergent_reference_figures.py`](../tools/plot_emergent_reference_figures.py) | Four-seed checkpoint replay, literal first-stimulus tuning baseline, seed aggregation, JSON, and reference-layout figures |
| [`figures/emergent_reference_comparison/all_alpha_assay_summary.json`](../figures/emergent_reference_comparison/all_alpha_assay_summary.json) | Portable compact copy of 14 existing assay metrics for every seed×alpha, plus four-seed mean/sample SEM and provenance |
| [`figures/emergent_reference_comparison/plot_data.json`](../figures/emergent_reference_comparison/plot_data.json) | Machine-readable values plotted from seeds 0–3 |
| [`figures/emergent_reference_comparison/`](../figures/emergent_reference_comparison/) | Two tuning panels, grouped decoding bars, and decoding/rate phase space |

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

## Four-seed protocol

- Development seeds: `0,1,2,3`.
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

This organization patch intentionally makes the reported six-alpha,
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
   `eps_rate=1e-8*36*R_ref`. Within one seed and arm, first compute the two
   condition means over all 216 final responses and all 36 L2/3 channels,
   `mu_A=mean_{p,i}(r_A[p,i])` and `mu_B=mean_{p,i}(r_B[p,i])`. Stored saving is
   the paired ratio of means
   `saving=(mu_B-mu_A)/(mu_B+eps_rate)`. The phase-space y coordinate uses the
   identical denominator and opposite numerator,
   `y=(mu_A-mu_B)/(mu_B+eps_rate)=-saving`. This is not the mean of 216
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
   profile, define `q_i=r_i/(sum_j r_j+eps_rate)`. `Cq` and `Fq` are the means
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

## Four-seed results

Values below are mean ± sample SEM over seeds `0,1,2,3`. `Delta` means
continuation A minus reversal B, except saving, which is `(B-A)/B`. `DeltaC`
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

### Displayed endpoints relative to the t0 baseline

Ratios here are computed from the across-seed mean displayed curves. Delta
uncertainties are seed SEM.

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
B)/reversal B)`. The endpoint values are:

- task-only endpoint (α=0.0): `Delta decode=.653537 ± .013624`,
  `(A-B)/B=-.076096 ± .001145`;
- rate-cost-weighted endpoint (α=0.9): `Delta decode=-.098995 ± .014371`,
  `(A-B)/B=-.542170 ± .019979`.

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
- Neural activity is metabolically constrained, but mean rate is still only a
  proxy here: Attwell & Laughlin (2001),
  [doi:10.1097/00004647-200110000-00001](https://doi.org/10.1097/00004647-200110000-00001),
  and Rangaraju, Calloway & Ryan (2014), *Activity-Driven Local ATP Synthesis Is
  Required for Synaptic Function*,
  [doi:10.1016/j.cell.2013.12.042](https://doi.org/10.1016/j.cell.2013.12.042).

## Exact RTX 5090 reproduction

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
Plotting branch B produces `plot_data.json` and:

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

- Four development seeds establish reproducibility for this software run, not
  confirmatory biological evidence.
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
- The rate-cost-weighted endpoint is broad attenuation with preferential center
  suppression, not absolute flank preservation.
- The plotted endpoints were selected coordinates (`alpha=0.0` and `0.9`), not
  evidence of a sharp phase boundary. The six-alpha series is descriptive.
- The plotter checks checkpoint schema and recomputes selected metrics, but it
  has no phenotype acceptance gate. The separate compact all-alpha artifact is
  derived from assay audit records rather than from `plot_data.json`.
- Standalone assay records do not store raw aligned profiles. Consequently the
  all-alpha artifact contains their scalar metrics, not reconstructed profiles;
  36-bin profiles in `plot_data.json` cover only the displayed endpoints.
- Exact reproducibility depends on compatible PyTorch, CUDA, driver, and GPU
  behavior; the repository does not pin a complete binary environment.
