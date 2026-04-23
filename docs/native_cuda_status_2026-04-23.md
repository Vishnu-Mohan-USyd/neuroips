# Native CUDA Status Report - 2026-04-23

## Scope

This document summarizes the current local code state in this worktree as of 2026-04-23. It is intended to accompany the current native CUDA/C++ migration work before commit/push.

It covers:

- the model architecture the repo is trying to implement
- the current native CUDA implementation surface
- the training and Richter assay flow
- what has been validated
- the current biological result status
- the strongest evidence-backed bottlenecks
- the next justified engineering step

This report reflects the local worktree contents, especially:

- `cpp_cuda/`
- `expectation_snn/cuda_sim/`
- `expectation_snn/validation/`
- `expectation_snn/assays/runtime.py`
- `scripts/diag_ctx_pred_richter_balance.py`

## High-Level Objective

The repo studies expectation effects in a sensory predictive-coding style SNN. The target outcome is that, after sequence learning, **expected trailers** produce a **dampened** sensory response relative to **unexpected trailers**, and that this can also be read out in a decoder / fMRI-like analysis.

The native CUDA work has two goals:

1. move the heavy simulation path off CPU so iteration is fast
2. preserve or recover the intended biological behavior from the Python/Brian2 specification

## Model Architecture

### V1 ring

The visual ring is organized into **12 orientation channels** at 15 degree spacing.

Per channel:

- `16` excitatory cells
- `4` SOM cells

Global inhibitory pool:

- `32` PV cells

Important V1 features from the source model:

- sensory drive enters V1 with a **Gaussian orientation profile**
- V1 excitatory cells have a **somatic** sensory drive path
- V1 excitatory cells also have an **apical** feedback path
- SOM provides channel-structured suppression
- PV provides broader stabilizing inhibition

Relevant source files:

- `expectation_snn/brian2_model/v1_ring.py`
- `expectation_snn/brian2_model/feedback_routes.py`

### Higher-area H system

The split architecture uses two higher-area rings:

- `H_context`
- `H_prediction`

Each ring follows the same 12-channel orientation structure, and the intended Brian2 implementation uses Wang-style recurrent dynamics with persistent bumps.

The intended functional roles are:

- `H_context`: hold the leader orientation across the leader-to-trailer delay
- `H_prediction`: express the forecast trailer state driven by learned `ctx -> pred` structure

Relevant source files:

- `expectation_snn/brian2_model/h_ring.py`
- `expectation_snn/brian2_model/h_context_prediction.py`

### Cross-area pathways

The key pathways are:

1. **Feedforward**
   - `V1_E -> H_context_E`
   - feature-matched Gaussian feedforward connectivity

2. **Plastic context-to-prediction mapping**
   - dense `192 x 192` `H_context -> H_prediction` matrix

3. **Predictive feedback to V1**
   - direct route: `H_prediction -> V1_E apical`
   - suppressive route: `H_prediction -> V1_SOM -> V1_E`

The intended feedback balance is controlled by:

- `g_total`
- `r = g_direct / g_som`

For the production split-path topology, the source model uses:

- direct route: center-only
- SOM route: wrapped surround over `±1` and `±2` channels

Relevant source files:

- `expectation_snn/brian2_model/feedforward_v1_to_h.py`
- `expectation_snn/brian2_model/feedback_routes.py`

## Training Structure

### Stage 0

Stabilize V1 inhibition, especially PV -> E, then freeze it.

Purpose:

- prevent runaway rates
- create a stable sensory substrate before sequence learning

### Stage 1

Learn `H_context -> H_prediction`.

Each training trial contains:

1. leader
2. trailer
3. ITI

Behavioral intent:

- leader drives V1 and `H_context`
- `H_context` persists through the delay
- trailer / teacher signal drives the correct `H_prediction`
- delayed modulatory gating updates `W_ctx_pred`

Important current production parameters used in the split-path work:

- `tau_coinc_ms = 500`
- `tau_elig_ms = 1000`
- `eta = 1e-3`
- `gamma = 1e-4`
- `w_target = 0.0075`
- `w_row_max = 3.0`

Relevant source files:

- `expectation_snn/brian2_model/train.py`
- `expectation_snn/cuda_sim/train_stage1_native.py`

## Richter Assay

The main assay under active native debugging is the deconfounded Richter-style expected/unexpected trailer test.

Current split-path interpretation:

- **expected**: trailer is the learned next step (`leader + 1`)
- **unexpected**: trailer is one of `leader + 2/3/4/5`

Default workload:

- `372` total trials

This is the main workload used for native GPU timing and biological-effect validation.

Relevant files:

- `expectation_snn/assays/richter_crossover.py`
- `scripts/diag_ctx_pred_richter_balance.py`
- `expectation_snn/cuda_sim/richter_native.py`

## Current Native CUDA Implementation

### Core native tree

Current native code lives in:

- `cpp_cuda/CMakeLists.txt`
- `cpp_cuda/include/expectation_snn_cuda/manifest.hpp`
- `cpp_cuda/src/bindings.cpp`
- `cpp_cuda/src/richter_eval.cu`

This now provides a standalone native execution path for the heavy simulation work.

### Python/native bridge

Current native support modules live in:

- `expectation_snn/cuda_sim/native.py`
- `expectation_snn/cuda_sim/export_bundle.py`
- `expectation_snn/cuda_sim/train_stage1_native.py`
- `expectation_snn/cuda_sim/richter_native.py`

### Validation surface

There is now a broad native validation layer under:

- `expectation_snn/validation/validate_native_*.py`

Examples include:

- neuron-kernel checks
- CSR scatter checks
- forward slice checks
- H dynamics checks
- ctx->pred plasticity checks
- checkpoint schema checks
- frozen Richter determinism/smoke checks
- native-vs-Brian comparison checks

## What Is Already Working

### Native executable path

A standalone native executable path exists and has been run successfully on the remote CUDA machine. The native runtime is no longer just a Python wrapper around CPU-heavy behavior.

### Stage 1 native checkpoint path

A native Stage 1 path exists and has produced accepted checkpoints in the current workflow.

### GPU speed

On the current validated full372 Richter workload:

- GPU runtime is on the order of `4 s`
- matched native CPU runtime is on the order of `216-228 s`
- observed speedup was approximately `50x`

This means the repo is now fast enough for short iteration cycles on the remote CUDA machine.

### Feedback-path correction work

The following native feedback-path repairs have already been made and verified as active in the current code path:

- source-faithful direct/SOM balance handling
- symmetric wrapped SOM surround over `±1` and `±2`
- corrected direct vs SOM route construction in the active production topology
- updated loop ordering so the suppressive SOM path is not trivially delayed relative to direct feedback

This matters because it rules out the earlier broken feedback-path semantics as the sole explanation for the current weak Richter effect.

## Current Biological Result Status

### Important distinction

The native runtime is now **fast**, but it is **not yet biologically acceptable** on the deconfounded Richter criterion.

The current effect is still in the **blip regime**, not the robust dampening regime.

### Latest measured native artifact state

Latest discussed native artifact:

- remote artifact name: `richter_dampening_seed42_default372_gpuonly_feedback_timingfix.json`

On that artifact:

- total V1 trailer rate/cell
  - expected: `16.4365 Hz`
  - unexpected: `16.4479 Hz`
  - delta: `-0.0114 Hz`

- target-channel V1 trailer rate/cell
  - expected: `196.8333 Hz`
  - unexpected: `196.9097 Hz`
  - delta: `-0.0764 Hz`

These are directionally correct for dampening, but far too small to count as a strong effect.

### Decoder status

The current native path still fails on decoding-style evidence:

- raw 12-channel decoder remains saturated
  - clean: `1.000 / 1.000`
  - noisy 8 Hz: `1.000 / 1.000`

- pseudo-voxel proxy remains near zero and not supportive of a meaningful fMRI-like dampening signature

So the current native path is still **NO-GO** for biological reproduction, even though the runtime is fast.

## What The Repo Actually Commits vs Documents

There are two different readout layers that need to be kept separate.

### Committed in code

- scalar Richter activity metrics
- pseudo-voxel forward model

### Documented in prose / current-state notes

- raw 12-channel V1 decoder
- `StandardScaler + multinomial LogisticRegression`
- leave-one-schedule-trial-out CV
- noisy readout using additive Gaussian measurement noise

Important historical note:

- the stronger `~ -1.25 Hz` Richter dampening band belongs to an **older confounded schedule**
- the current deconfounded split-path target is weaker and is documented mainly through the later post-Kuhn/split-path notes

That means the current native path should not claim success merely because the sign is negative. It must be judged against the deconfounded split-path expectations, not the old adaptation-confounded regime.

## Current Root-Cause Picture

The current debugging evidence points to two main remaining issues, but they are not equally strong.

### 1. Strongest currently confirmed active mismatch: stimulus geometry

This is the clearest current mismatch between the active native Richter path and the Brian2 specification.

Brian2 V1 source model:

- sensory drive is Gaussian across orientation channels
- `sigma_stim_deg = 22.0`

Active CUDA Richter path:

- stimulus generation is still effectively **single-channel plus baseline**

Why this matters:

- the current V1 trailer response is almost exactly one-hot
- bins 1-4 have exactly one active channel per trial
- second-highest channel activity is effectively zero
- decoder margins are therefore enormous and saturated

This is the strongest direct explanation for the present one-hot V1 geometry.

### 2. Real architectural gap, but second in the current ranking: H recurrence consumption

The intended split-path architecture relies on recurrent H dynamics, especially for `H_context`.

Important current state:

- the standalone native production checkpoint path does **not** currently load committed checkpointed `ctx_ee` / `pred_ee` banks
- the active Richter production loop does **not** use those checkpointed recurrent structures
- there are hardcoded H recurrent kernels elsewhere in CUDA code, but not in the active frozen-Richter production path under discussion

Why this matters:

- it is a real architecture mismatch
- it plausibly weakens H persistence / forecast quality
- but it is not the strongest direct explanation for the one-hot V1 geometry

### 3. What has already been ruled down

The following are no longer the leading explanation:

- old feedback route asymmetry alone
- old direct/SOM balance bug alone
- pure schedule mixing / windowing artifact alone

Those surfaces were important to fix, but the latest evidence says they do not explain the remaining tiny magnitude by themselves.

## Current Best-Supported Next Step

The next smallest justified intervention is:

1. patch the active native Richter stimulus generator to match the Brian2 Gaussian sensory geometry
2. rerun full372 GPU
3. check whether:
   - V1 geometry broadens
   - the second channel rises above zero
   - margins drop
   - decoder saturation weakens
   - dampening magnitude grows materially

If that fails, the next larger intervention is:

- wire checkpointed H recurrence into the active native Richter path

This ordering is important because:

- Gaussian stimulus geometry is a smaller, cleaner, directly observed mismatch
- H recurrence wiring is a larger architectural change

## Expected Decision Logic After The Next Run

### If Gaussian geometry works

Expected signatures:

- V1 trailer is no longer strictly one-hot
- the second channel and off-target mass become nontrivial
- decoder saturation weakens
- effect size increases materially

Then:

- keep the geometry patch
- reassess whether recurrence is still needed for additional magnitude

### If Gaussian geometry fails

Expected signatures:

- V1 broadens somewhat, but dampening remains tiny
- H expected/unexpected separation remains weak

Then:

- H recurrence becomes the next justified intervention

## Current Repository Delta To Be Committed

At the time this report was written, the worktree includes:

- modified tracked files such as:
  - `expectation_snn/assays/runtime.py`
  - `expectation_snn/brian2_model/h_context_prediction.py`
  - `expectation_snn/brian2_model/h_ring.py`
  - `expectation_snn/scripts/smoke_test.py`
  - `expectation_snn/validation/validate_feedback_routes.py`
  - `scripts/diag_ctx_pred_richter_balance.py`

- untracked new native/runtime trees such as:
  - `cpp_cuda/`
  - `expectation_snn/cuda_sim/`
  - many `expectation_snn/validation/validate_native_*.py` files

This means the current branch now contains both:

- the native execution surface
- the validation harness around it

## Current Bottom Line

The repo is no longer blocked on CPU runtime. The native CUDA path is fast enough for iterative debugging.

However, the current native Richter result is still biologically weak:

- correct sign only
- tiny magnitude
- saturated decoder
- one-hot V1 geometry

The current best evidence says the next fix should target the active sensory-geometry mismatch before attempting the larger H-recurrence rewrite in the standalone native Richter path.
