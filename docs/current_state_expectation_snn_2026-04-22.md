# Expectation SNN Current State - 2026-04-22

This document records the current code and evidence state after the bottom-up
debugging pass on the split `H_context / H_prediction` expectation SNN.

It is intentionally explicit. The project is not in a final scientific state.
The code now has a working bounded Stage-1 split-architecture checkpoint and
diagnostic evidence for expectation-linked dampening and local profile
sharpening, but the full `n=360` Stage-1 retrain fails its forecast gate and
Stage-2 cue learning is not implemented for the split architecture.

## Repository State

Base branch at the time of this work:

```text
expectation-snn-v1h
```

Main code paths touched:

```text
expectation_snn/brian2_model/h_context_prediction.py
expectation_snn/brian2_model/train.py
expectation_snn/brian2_model/feedback_routes.py
expectation_snn/assays/runtime.py
expectation_snn/validation/validate_h_context_prediction.py
expectation_snn/validation/validate_feedback_routes.py
expectation_snn/validation/validate_runtime_ctx_pred.py
scripts/train_stage1_ctx_pred_full.py
```

New validation scripts:

```text
expectation_snn/validation/validate_ctx_pred_delayed_gate_learning.py
expectation_snn/validation/validate_ctx_pred_h_context_gate_persistence.py
expectation_snn/validation/validate_h_context_persistence.py
expectation_snn/validation/validate_v1_inhibitory_motif.py
```

New diagnostic scripts:

```text
scripts/diag_ctx_pred_richter_balance.py
scripts/diag_ctx_pred_timing_matrix.py
scripts/diag_h_persistence_clamp.py
```

Generated checkpoints, logs, CUDA smoke outputs, and JSON diagnostics are local
artifacts under `data/` and `logs/`. They are referenced below but are not
intended to be treated as source code.

## Scientific Question

The model asks whether expectation feedback into V1 produces:

1. Dampening: expected stimuli evoke weaker V1 responses and/or weaker decoding
   evidence than unexpected stimuli.
2. Sharpening: expected stimuli evoke a more selective response profile, with
   stronger preferred-channel response and weaker local flanks.

The current positive evidence is for a Richter-style sensory-context
expectation:

```text
leader stimulus -> H_context memory -> H_prediction forecast -> V1 feedback
```

It is not yet evidence for Kok-style arbitrary cue expectation, because Stage-2
cue learning is still legacy-only and does not yet support the split
architecture.

## Architecture

### V1 Ring

V1 is a 12-channel orientation ring over the 180 degree orientation space.
Channels are spaced every 15 degrees.

Each V1 channel contains:

- excitatory LIF neurons,
- co-tuned SOM interneurons,
- broad/shared PV stabilization.

V1 excitatory neurons have a somatic voltage and an apical/modulatory current
pathway. Sensory stimulus drive enters the somatic pathway. Direct top-down
feedback enters the apical/modulatory pathway and is not intended to drive
spikes alone.

### H_context / H_prediction Split

The higher-area model is split into two rings:

```text
H_context     stores current/recent sensory context
H_prediction  represents expected next stimulus
```

`H_context` is a recurrent NMDA-supported ring intended to hold the leader
orientation across the leader-to-trailer transition. `H_prediction` is driven
by a learned all-to-all `H_context -> H_prediction` transform plus trailer
teacher input during Stage-1 training.

The core learned transformation is:

```text
leader channel in H_context -> expected trailer channel in H_prediction
```

### Feedback Routes

The model uses two feedback routes from `H_prediction` to V1.

1. Direct route:

```text
H_prediction -> V1_E apical/modulatory current
```

Current ctx_pred default topology:

```text
center-only feature-matched direct feedback
```

2. SOM route:

```text
H_prediction -> V1_SOM -> V1_E inhibition
```

Current ctx_pred default topology:

```text
d1/d2 local-surround SOM feedback

for predicted channel c:
  SOM c +/- 1 gets 0.4 each
  SOM c +/- 2 gets 0.1 each
  SOM c gets 0.0
```

This is fixed and label-blind. The motivation is that the earlier co-tuned
Gaussian direct and co-tuned Gaussian SOM feedback could produce co-tuned gain
or suppression but could not produce strict center-up/local-flank-down
sharpening.

### Feedback Balance Parameter

The balance parameter is:

```text
r = g_direct / g_SOM
```

Total feedback is held fixed by the code.

Interpretation:

```text
r = 0       pure SOM-route ablation/control
r = 0.25    SOM-heavy feedback
r = 4       direct-heavy feedback
r = inf     pure direct-route ablation/control
```

Biological mapping:

- direct route approximates top-down input to pyramidal apical dendrites,
- SOM route approximates top-down recruitment of dendrite-targeting
  SOM/Martinotti inhibition and local surround suppression,
- fixed total feedback is a model control, not a known biological conservation
  law.

## Training Stages

### Stage 0

Stage 0 calibrates the V1 ring. It loads/stores V1 bias currents and PV weights.
This is an engineering stabilization step, not a scientific result.

Relevant checkpoint:

```text
stage_0_seed42.npz
```

### Stage 1: Context-to-Prediction Learning

Stage 1 trains the split architecture under Richter-like leader-trailer
sequences.

Sequence logic:

```text
leader appears
H_context receives leader drive
H_prediction teacher is off

trailer appears
H_context external drive is off, so it must persist leader internally
H_prediction receives trailer teacher drive

after trailer response
modulatory gate updates H_context -> H_prediction weights

ITI
H_context is quiesced
ctx->pred transmission is temporarily disabled, then restored before next trial
```

The schedule is biased, not balanced. Each leader has a statistically expected
trailer. That creates a learnable transition structure.

### Stage 2: Cue Learning

Current status:

```text
not implemented for the split ctx_pred architecture
```

The existing Stage-2 cue-learning code is legacy single-ring `H_R` only. It
expects:

```text
stage_1_hr_seed42.npz
```

and loads a single H ring with an `ee_w_final` key. It cannot currently load:

```text
stage_1_ctx_pred_seed42.npz
```

which stores:

```text
ctx_ee_w_final
pred_ee_w_final
W_ctx_pred_final
```

Runtime also explicitly rejects cue loading under:

```text
architecture = "ctx_pred"
with_cue = True
```

The likely correct split-architecture Stage-2 design is to train cue input onto
`H_context`, not directly onto `H_prediction`, so that cue-driven context can
flow through the learned `H_context -> H_prediction` transform.

## Major Failure Modes Found And Fixed

### 1. Stage-1 Gate Was Too Early

Original failure:

```text
M-gate consumed eligibility at trailer onset.
H_prediction trailer spikes had not happened yet.
Leader-copy links were strengthened more than leader->trailer links.
```

Fix:

```text
ctx_pred M-gate now occurs at trailer offset/end.
```

Validation:

```text
validate_ctx_pred_delayed_gate_learning: PASS
```

### 2. Nonzero Random ctx->pred Init Caused Leader Copying

Original failure:

```text
random ctx->pred weights made H_prediction fire during leader windows.
that produced leader-copy eligibility before the trailer teacher.
```

Fix:

```text
w_init_frac = 0.0 for the production ctx_pred training config
```

### 3. H_context Was Overwritten By Trailer

Original failure:

```text
at the delayed gate, H_context represented actual trailer, not leader.
learning reinforced trailer/self associations.
```

Fix:

```text
during trailer:
  V1->H_context off
  H_context cue/input off
  H_prediction teacher on
```

Validation:

```text
H_context at gate: leader_match = 1.0 in bounded passing runs
```

### 4. ITI Residual H_context Drove H_prediction

Original failure:

```text
H_context residual activity during ITI drove H_prediction through learned ctx->pred.
H_prediction trial-start state contaminated preprobe forecasts.
```

Fix:

```text
after a short trailer-end gate flush:
  quiesce H_context during ITI
  temporarily set ctx->pred transmission to zero during ITI tail
  restore ctx->pred before next leader/trailer
```

Validation:

```text
ITI H_context rate = 0 Hz
ctx->pred disabled windows = number of ITIs
ctx->pred restored = true
```

### 5. Stage-1 Persistence Metric Measured The Wrong Thing

Original failure:

```text
the saved "H bump persistence" metric measured H_prediction.
ctx->pred remained active, so H_context kept driving H_prediction during the decay window.
```

Fix:

```text
ctx_pred Stage-1 pass criterion uses H_context persistence.
H_prediction autonomous persistence is recorded as a diagnostic only.
```

### 6. Training Normalizers Were Active During Validation Probes

Original failure:

```text
recurrent postsynaptic normalizers kept running during post-training probes.
this artificially inflated persistence to 990 ms.
```

Fix:

```text
disable recurrent normalizers during post-training validation probes.
```

### 7. Original Feedback Topology Could Not Produce Strict Sharpening

Original topology:

```text
H_prediction -> V1_E: Gaussian co-tuned
H_prediction -> V1_SOM: Gaussian co-tuned
V1_SOM -> V1_E: same-channel inhibition
```

That topology can produce:

```text
co-tuned suppression or co-tuned gain
```

but not:

```text
center-up / local-flank-down sharpening
```

Fix:

```text
ctx_pred direct route: center-only
ctx_pred SOM route: d1/d2 local surround
```

Validation:

```text
validate_feedback_routes: PASS
```

### 8. H_prediction Was Too Broad

Original failure:

```text
broad H_prediction population caused multiple feedback source channels to lift V1 flanks.
even center-only direct feedback could not sharpen if H_prediction itself was broad.
```

Final bounded production candidate:

```text
prediction ring:
  w_inh_e_init = 1.0
  inh_w_max = 3.0
pred_e_uniform_bias_pA = 100.0
drive_amp_ctx_pred_pA = 400.0
w_init_frac = 0.0
```

This is set in:

```text
scripts/train_stage1_ctx_pred_full.py
```

## Current Positive Bounded Result

Fresh bounded checkpoint:

```text
data/checkpoints_ctxpred_n72_prod_postkuhn_seed42_final/stage_1_ctx_pred_seed42.npz
```

Stage-1 n=72 passed:

```text
h_bump_persistence_ms = 470.0 ms
h_preprobe_forecast_prob = 0.583333
no_runaway = 34.463889 Hz
ctx_gate_leader_match_frac = 1.0
ctx_gate_trailer_match_frac = 0.0
```

Final no-override downstream assay:

```text
data/diag_ctx_pred_richter_balance_n72_prod_postkuhn_nooverride_seed42_n36_r025r4.json
```

The assay loaded checkpoint metadata:

```text
ctx_pred_config_source = checkpoint
drive_amp_ctx_pred_pA = 400.0
pred_e_uniform_bias_pA = 100.0
feedback direct kernel = center
feedback SOM kernel = d1_d2_surround
```

No manual drive/bias override was used.

## Expected vs Unexpected Results

The primary comparison is expected vs unexpected trailer response.

### Clean MVPA Decoder

Artifact:

```text
data/diag_prod_postkuhn_mvpa_decoder_expected_unexpected_seed42_n36.json
```

Decoder:

```text
features = 12-channel V1_E rate vector per trial
model = StandardScaler + multinomial LogisticRegression
CV = leave-one schedule trial index out
```

Clean decoding accuracy saturated:

```text
feedback off:  expected = 1.000, unexpected = 1.000
r=0.25:        expected = 1.000, unexpected = 1.000
r=4:           expected = 1.000, unexpected = 1.000
```

This means the clean visual task is too easy for decoding accuracy to be a
central measure.

### Noisy MVPA Decoder

Artifact:

```text
data/diag_prod_postkuhn_mvpa_decoder_expected_unexpected_gauss8hz_seed20260422_n200.json
```

Manipulation:

```text
additive Gaussian V1 channel measurement noise
sigma = 8 Hz
clipped at 0 Hz
200 seeded replicates
no retraining
```

This was the first tested noise level that reliably dropped accuracy below
ceiling while keeping performance above chance.

Expected vs unexpected decoding:

```text
feedback off:
  expected = 0.915
  unexpected = 0.922

r=0.25:
  expected = 0.757
  unexpected = 0.862

r=4:
  expected = 0.919
  unexpected = 0.921
```

Interpretation:

```text
r=0.25 gives the expected dampening signature:
  expected decoded worse than unexpected

r=4 does not improve decoding accuracy:
  sharpening evidence is response-shape based, not decoder-accuracy based
```

### Activity / Energy

Under noisy analysis, expected minus unexpected:

```text
r=0.25:
  population rate lower
  total rate lower
  peak/center lower
  squared energy lower

r=4:
  center higher
  near flank lower
  total/population activity close to unchanged or slightly lower
```

### Response Shape

Expected minus unexpected, aligned to actual stimulus channel:

```text
r=0.25:
  center down
  near flanks down
  activity/energy down
  classification: dampening/suppression

r=4:
  center up
  near flanks down
  total activity not broadly increased
  classification: local response-profile sharpening
```

Important caveat:

```text
r=4 sharpening is local profile sharpening.
It is not improved decoding accuracy.
It is not yet broad decile-wide surround suppression.
```

## Full n=360 Stage-1 Retrain

Full retrain was run with current production config.

Command log:

```text
logs/n360_prod_postkuhn_seed42_full_20260422T050105Z.log
```

Checkpoint:

```text
data/checkpoints_ctxpred_n360_prod_postkuhn_seed42_full/stage_1_ctx_pred_seed42.npz
```

Diagnosis:

```text
data/diag_stage1_n360_failure_checkpoint_summary_seed42.json
```

Result:

```text
FAILED saved Stage-1 gate
```

Pass/fail components:

```text
h_bump_persistence_ms = 210.0 ms      PASS
h_preprobe_forecast_prob = 43/180     FAIL, 0.2389 < 0.25
no_runaway = 24.07 Hz                 PASS
```

The failure is forecast/readout degradation, not persistence or runaway.

n72 vs n360:

```text
n72 forecast tail = 0.5833
n360 forecast tail = 0.2389

n72 W row peak expected count = 5/6
n360 W row peak expected count = 3/6

n72 desired-copy margin = 0.03724
n360 desired-copy margin = 0.01264
```

n360 H_prediction argmax collapsed mainly to channels:

```text
[1, 2, 3]
```

No Stage-2 or downstream assay was run from the failed n360 checkpoint.

## Stage-2 Cue Learning Status

Stage-2 cue learning is currently not valid for the split architecture.

Current code path:

```text
run_stage_2_cue(...)
```

builds:

```text
single H_R ring
```

and expects:

```text
stage_1_hr_seed42.npz
ee_w_final
```

The split ctx_pred checkpoint instead provides:

```text
stage_1_ctx_pred_seed42.npz
ctx_ee_w_final
pred_ee_w_final
W_ctx_pred_final
```

Runtime also explicitly blocks:

```text
architecture = "ctx_pred"
with_cue = True
```

because cue synapses are only implemented for the legacy `h_r` runtime path.

Needed implementation:

1. Add ctx_pred-aware Stage-2 mode.
2. Load `stage_1_ctx_pred_seed42.npz`.
3. Train cue synapses onto `H_context`, not directly onto `H_prediction`.
4. Save Stage-2 metadata:

```text
architecture = ctx_pred
cue_target = H_context
```

5. Update runtime so `with_cue=True` attaches cue weights to
   `bundle.ctx_pred.ctx`.
6. Validate that cue-driven `H_context` produces `H_prediction` forecast via
   learned `W_ctx_pred`.

## Validation Commands That Passed

Representative commands:

```bash
mamba run -n expectation_snn python -m expectation_snn.validation.validate_h_context_prediction
mamba run -n expectation_snn python -m expectation_snn.validation.validate_ctx_pred_delayed_gate_learning
mamba run -n expectation_snn python -m expectation_snn.validation.validate_ctx_pred_h_context_gate_persistence
mamba run -n expectation_snn python -m expectation_snn.validation.validate_h_context_persistence
mamba run -n expectation_snn python -m expectation_snn.validation.validate_feedback_routes
mamba run -n expectation_snn python -m expectation_snn.validation.validate_richter_biased_schedule
```

Known validator blocker:

```text
validate_runtime_ctx_pred
```

The full module is blocked in this checkout because default legacy checkpoints
are missing under `expectation_snn/data/checkpoints`. Targeted ctx_pred runtime
checks passed using:

```text
data/checkpoints_ctxpred_n72_prod_postkuhn_seed42_final
```

## Current Interpretation

What is supported:

```text
bounded n=72 sensory-context expectation can produce:
  SOM-heavy dampening
  direct-heavy local response-profile sharpening
```

The dampening result is strongest:

```text
under noisy MVPA readout:
  expected decoded worse than unexpected
  expected activity lower
  expected peak lower
```

The sharpening result is narrower:

```text
r=4:
  expected center/preferred response higher
  near flanks lower
  decoding accuracy not improved
```

What is not yet supported:

```text
full n=360 robust Stage-1 success
Stage-2 cue-driven expectation in split architecture
Kok-style arbitrary cue expectation
Tang held-out rotating-sequence result
broad decile-wide sharpening
decoder-accuracy improvement for direct-heavy expectation
```

## Immediate Next Technical Work

1. Diagnose why n360 degrades after n72.
   Likely area: long-run W_ctx_pred drift / H_prediction readout collapse.

2. Implement split-architecture Stage-2 cue learning.
   Cue should drive `H_context`; `H_prediction` should arise through learned
   `W_ctx_pred`.

3. Make the sensory decoding task intrinsically harder.
   The current clean V1 representation is too separable. Noisy analysis-time
   decoding is useful, but a better scientific assay should introduce
   controlled stimulus ambiguity/noise in the actual simulation.

4. Re-run no-override assays only from checkpoints that pass the relevant
   training gate.
