# Phase 6 Report: Experimental Paradigms

**Status:** All 5 paradigms implemented. 42 new tests, 274/274 total pass.

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `src/experiments/paradigm_base.py` | 170 | Base class: TrialConfig, TrialSet, ConditionData, ExperimentResult, batched execution |
| `src/experiments/hidden_state.py` | 92 | P1: CW/CCW/neutral with dense probe offsets |
| `src/experiments/omission.py` | 62 | P2: 10 context + blank at position 11 |
| `src/experiments/ambiguous.py` | 86 | P3: mixture and low-contrast probes |
| `src/experiments/task_relevance.py` | 64 | P4: task_state manipulation |
| `src/experiments/surprise_dissociation.py` | 72 | P5: CW-unexpected vs CCW-expected |
| `src/experiments/__init__.py` | 20 | ALL_PARADIGMS registry |
| `scripts/run_experiments.py` | 68 | CLI entry point |
| `tests/test_experiments.py` | 232 | 42 tests |

## Paradigm Summary

### P1: Hidden-State Transitions (core paradigm)

- **Conditions**: 2 rules (CW/CCW) x 6 deviations (0/15/30/45/60/90) + 6 neutral = 18
- **Trial structure**: 8 context (steps_on=8, steps_isi=4) + probe + post = 112 timesteps
- **Probe onset**: timestep 96
- **Neutral matching**: last context stimulus matched, preceding context randomized
- **Dense offsets**: 0, 15, 30, 45, 60, 90 deg (configurable)

### P2: Omission Trials

- **Conditions**: 2 rules x 2 types (omission/present) = 4
- **Trial structure**: 10 context + probe/blank + extended post = 140 timesteps
- **Omission**: probe_contrast=0.0 produces zero stimulus
- **Extra window**: "omission" covering timesteps 120-140
- **Key test**: deep template decoding during blank period

### P3: Ambiguous Stimuli

- **Conditions**: 2 rules x 3 types (mixture/low_contrast/clear) = 6
- **Mixture**: 50/50 blend of expected +/- 15 deg (uses `make_ambiguous_stimulus`)
- **Low contrast**: c=0.15 at expected orientation
- **Clear**: full contrast control

### P4: Task Relevance

- **Conditions**: 2 tasks (relevant/irrelevant) x 2 (expected/unexpected) = 4
- **Relevant**: task_state = [1, 0] for all timesteps
- **Irrelevant**: task_state = [0, 1] for all timesteps
- **No re-training**: same frozen model, different task_state input to V2

### P5: Surprise Dissociation

- **Conditions**: 2 (high_v2_surprise, low_v2_surprise)
- **Design**: CW context vs CCW context, both ending at same theta_L
- **Probe**: theta_L - step (= CCW prediction, 30 deg from CW prediction)
- **Matched**: same probe, same last context, same local distance (step)
- **Dissociated**: V2 surprise (high for CW-unexpected, low for CCW-expected)
- **V2 entropy**: recorded via state_logits trajectory for analysis-time validation

## Trajectory Shapes (sample: 3 trials)

All paradigms record 8 trajectory fields per condition per trial:

| Field | P1 Shape | P2 Shape | P3-P5 Shape |
|-------|----------|----------|-------------|
| r_l23 | [3, 112, 36] | [3, 140, 36] | [3, 112, 36] |
| r_l4 | [3, 112, 36] | [3, 140, 36] | [3, 112, 36] |
| r_pv | [3, 112, 1] | [3, 140, 1] | [3, 112, 1] |
| r_som | [3, 112, 36] | [3, 140, 36] | [3, 112, 36] |
| q_pred | [3, 112, 36] | [3, 140, 36] | [3, 112, 36] |
| pi_pred | [3, 112, 1] | [3, 140, 1] | [3, 112, 1] |
| state_logits | [3, 112, 3] | [3, 140, 3] | [3, 112, 3] |
| deep_template | [3, 112, 36] | [3, 140, 36] | [3, 112, 36] |

## Temporal Windows

| Window | P1/P3-P5 | P2 |
|--------|----------|-----|
| prestimulus | (92, 96) | (116, 120) |
| early | (96, 99) | (120, 123) |
| sustained | (99, 104) | (123, 128) |
| late | (104, 112) | (128, 140) |
| omission | n/a | (120, 140) |

## Validation Checks

1. P1: 18 conditions, correct trial counts, dense probe offsets present
2. P2: omission probe stimulus = zero, present probe nonzero
3. P3: mixture probe nonzero, low-contrast energy < clear energy
4. P4: task_state [1,0] for relevant, [0,1] for irrelevant, constant across trial
5. P5: matched probe and last context (torch.allclose), different earlier context
6. All: trajectories finite, temporal windows within bounds, consistent shapes

## Test Results

```
$ python -m pytest tests/test_experiments.py -v
42 passed in 426.48s

$ python -m pytest tests/ -q
274 passed in 566.81s
```
