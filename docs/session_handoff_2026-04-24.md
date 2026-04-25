# Session Handoff - 2026-04-24

This file is the restart point for the current multi-agent investigation.

## Repository / branch

- Repo: `/home/vysoforlife/code_files/snn_v2_gpu/neuroips`
- Current pushed branch: `native-cuda-status-20260423`
- Earlier status doc: [native_cuda_status_2026-04-23.md](/home/vysoforlife/code_files/snn_v2_gpu/neuroips/docs/native_cuda_status_2026-04-23.md)

## Remote machine

- SSH: `ssh root@38.65.239.32 -p 18963 -i ~/.ssh/id_ed25519`
- Remote repo: `/workspace/neuroips_gpu_migration_20260422/neuroips`
- Remote env:
  - `source /workspace/neuroips_gpu_migration_20260422/cuda_env.sh`
  - `source /workspace/neuroips_gpu_migration_20260422/.venv_brian2cuda/bin/activate`
- Logs dir: `/workspace/neuroips_gpu_migration_20260422/logs`

## Current investigation policy

The user explicitly required:

- bottom-up, incremental debugging
- no architecture changes unless a concrete failing link is isolated first
- no status-only chatter
- only speak when there is an evidence-bearing result or a real blocker
- always wait on agents
- compute should run remotely

The current ladder is:

1. sensory stage
2. higher-area temporal learning
3. energy/loss effects
4. feedback / expected-vs-unexpected dampening

We are **still in Phase 2**. Do not jump to energy or feedback yet.

## Local worktree state at handoff

`git status --short` at handoff:

- modified:
  - `cpp_cuda/include/expectation_snn_cuda/manifest.hpp`
  - `cpp_cuda/src/richter_eval.cu`
- untracked:
  - `scripts/analyze_native_sensory_diagnostics.py`
  - `scripts/analyze_native_stage1_temporal_learning.py`
  - `scripts/__pycache__/`

## Phase 1 result

### Sensory-stage diagnosis

Initial sensory stage failed because the native front end was overdriven and trivially separable.

Confirmed by sweep:

- rates tested: `50, 100, 150, 200, 300, 500`
- key result: `100 Hz` was the only tested point that broke noisy `V1_E` saturation without collapsing tuning

Remote artifacts:

- sweep dir:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/sensory_sweep_20260424`

Accepted Phase 1 operating point:

- `grating-rate-hz = 100`

Archimedes verdict:

- `GO` for Phase 1 at `rate100`

Important note:

- this was only a sensory-stage pass
- it does **not** imply expectation dampening is solved

## Phase 2 result so far

### First Phase 2 failure

The native Stage1 trainer initially regressed with longer training.

Observed in:

- `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424/stage1_temporal_learning_summary_seed42.json`

Key regression:

- `forecast_probability`: `1.0000 -> 0.6111` from `n=6 -> n=72`
- `expected_argmax_accuracy`: `1.0000` early, then degraded to `0.6667`

Hooke localized cause:

- recency-dominated overwrite in generated Stage1 trainer
- trainer followed actual trailer per leader, not a stable expected target

### Phase 2 fix 1

Patched generated trainer to use the schedule's `expected_trailer_idx`.

Result:

- removed the regression
- but Phase 2 still failed because the evaluation metric was not truly held out

Artifacts:

- fixed sweep:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix1`
- summary:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix1/stage1_temporal_learning_summary_seed42.json`

### Phase 2 measurement fix

Added a true held-out native evaluation command:

- `stage1-heldout-eval`

This runs an independent generated schedule from a different seed and records `H_pred` pretrailer counts.

Artifacts:

- held-out eval dir:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix1/heldout_eval_seed4242`
- held-out summary:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_sweep_20260424_fix1/stage1_temporal_learning_summary_seed42_heldout_seed4242.json`

### True held-out Phase 2 failure

Even after the held-out evaluation fix, Phase 2 still failed:

- held-out expected argmax stayed at ceiling
- held-out forecast was modest and noisy, not convincingly improving

Archimedes verdict:

- `NO-GO — measurable training improvement`

### Phase 2 fix 2

Tried narrowing the expected post teacher from channel-wide to single-cell while keeping stable expected target.

Result:

- scientifically a no-op
- fix1 and fix2 checkpoints / held-out payloads were bit-identical

Hooke confirmed no-op.

### Discriminating run after fix 2

Changed the pre teacher from channel-wide to single-cell while keeping single-cell expected post teacher.

This produced a **real scientific divergence** at `n=6`:

- checkpoint hash changed
- `W_ctx_pred` hash changed
- held-out payload changed
- held-out counts changed

Artifacts:

- discriminant dir:
  - `/workspace/neuroips_gpu_migration_20260422/neuroips/expectation_snn/data/checkpoints_native_cpp/stage1_temporal_discriminant_20260424`
- logs:
  - `/workspace/neuroips_gpu_migration_20260422/logs/native_stage1_discriminant_build_20260424.log`
  - `/workspace/neuroips_gpu_migration_20260422/logs/native_stage1_discriminant_run_n6_20260424.log`

But Hooke found:

- this changed only **scale**
- it did **not** change the core held-out confusion pattern
- held-out `row_argmax`, `expected_argmax_accuracy`, `actual_argmax_accuracy`, and `forecast_probability` were unchanged

So the discriminating pre-surface patch was real, but it did not move Phase 2 in the right direction.

## Last confirmed Hooke conclusion

The next remaining first cause after the discriminating run:

- the changed training surface moved only magnitude/scale
- the core held-out mapping stayed identical
- therefore a full `n={6,12,24,48,72}` sweep is **not justified yet**

Hooke’s exact recommendation:

- change the remaining common target-construction surface at:
  - `cpp_cuda/src/native_cli.cpp:1679-1692`
- keep the current single-cell pre/post event surface in:
  - `cpp_cuda/src/richter_eval.cu:9672-9688`
- rerun only the same discriminating `n=6` train + held-out eval
- check whether:
  - `row_argmax`
  - `expected_argmax_accuracy`
  - `actual_argmax_accuracy`
  - `forecast_probability`
  finally diverge from the current `fix2` pattern

## In-flight task when the session stopped

This exact task was handed to Fermat:

- modify only the target-construction surface in `native_cli.cpp:1679-1692`
- do **not** start a full sweep
- rerun only discriminating `n=6`
- compare against current fix2 pattern

At handoff, no completed result from that task had been received.

Important:

- the wait on Fermat was interrupted by the user
- the remote work may still be running
- first action after resume should be to check whether Fermat’s last remote `n=6` run completed and whether any new artifact/log landed

## What not to do on resume

- do not jump to energy/loss analysis yet
- do not jump to feedback/dampening yet
- do not run a full Phase 2 sweep until the next `n=6` discriminating run actually changes the core held-out pattern
- do not claim progress from sign-only or scale-only changes
- do not speak to the user unless there is an evidence-bearing result or a real blocker

## Immediate next step on resume

1. Check whether the last Fermat `n=6` target-construction run completed.
2. If it completed:
   - compare the new checkpoint and held-out artifact against `fix2`
   - ask Hooke whether the core held-out pattern changed
3. If it did not complete:
   - rerun only that same `n=6` discriminating experiment
4. Only if the core held-out pattern changes:
   - decide whether a full Phase 2 sweep is now justified
