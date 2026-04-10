# Phase 1 Baseline Snapshot — Pre-SNN Port

**Captured by:** Validator
**Date:** 2026-04-10
**Purpose:** Freeze the rate-model state at the moment just before SNN code lands, so regressions can be detected precisely.

---

## Git State

| Field | Value |
|---|---|
| Branch | `feature/snn-port` |
| HEAD SHA | `59b6c161ed32f5d6c8b2881cef698e0887a7fe2e` |
| HEAD SHA (short) | `59b6c16` — **matches the expected baseline** |
| HEAD message | "Rewrite RESULTS.md and ARCHITECTURE.md with complete 25-run sweep findings" |
| Working tree | Clean (no uncommitted changes) |

## Test Suite Baseline

Command: `python -m pytest tests/ -q --tb=no`

| Metric | Value |
|---|---|
| Tests collected | 374 |
| Passed | 374 |
| Failed | 0 |
| Errored | 0 |
| Warnings | 14 (all `torch.jit.script_method` deprecation in `tests/test_training.py`) |
| Wall time | ~16.2s |

**Regression rule for Phase 1:** any Phase 1 commit that lowers the pass count below 374 on the rate model test files is an automatic NO-GO.

## Source File Inventory — `src/`

32 Python files across 7 subpackages. This is the "rate-model only" footprint; the SNN port must add `src/spiking/` without mutating these files (except `src/config.py` per plan).

```
src/__init__.py
src/config.py                              <- will gain SpikingConfig (Phase 1.7)
src/state.py                               <- stays (rate NetworkState)
src/utils.py
src/model/__init__.py
src/model/populations.py
src/model/v2_context.py
src/model/feedback.py
src/model/network.py
src/stimulus/__init__.py
src/stimulus/gratings.py
src/stimulus/sequences.py
src/training/__init__.py
src/training/losses.py
src/training/stage1_sensory.py
src/training/stage2_feedback.py
src/training/trainer.py
src/analysis/__init__.py
src/analysis/ablations.py
src/analysis/bias_analysis.py
src/analysis/decoding.py
src/analysis/energy.py
src/analysis/feedback_discovery.py
src/analysis/model_recovery.py
src/analysis/observation_model.py
src/analysis/omission_analysis.py
src/analysis/plotting.py
src/analysis/rsa.py
src/analysis/suppression_profile.py
src/analysis/temporal_analysis.py
src/analysis/tuning_curves.py
src/analysis/v2_probes.py
src/experiments/__init__.py
src/experiments/ambiguous.py
src/experiments/hidden_state.py
src/experiments/omission.py
src/experiments/paradigm_base.py
src/experiments/surprise_dissociation.py
src/experiments/task_relevance.py
```

## Test File Inventory — `tests/`

```
tests/__init__.py
tests/test_analysis.py
tests/test_experiments.py
tests/test_model_forward.py
tests/test_model_recovery.py
tests/test_network.py
tests/test_stimulus.py
tests/test_training.py
```

Plan expects three new test files in Phase 1.8/1.9:
- `tests/test_spiking_populations.py`
- `tests/test_spiking_network.py`
- `tests/test_spiking_v2.py`

## Expected New Files After Phase 1

Per plan (`~/.claude/plans/quirky-humming-giraffe.md` lines 60-73), Phase 1 is allowed to add exactly these files and no others:

```
src/spiking/__init__.py
src/spiking/surrogate.py          (Task #11, Phase 1.1)
src/spiking/filters.py            (Task #12, Phase 1.2)
src/spiking/state.py              (Task #13, Phase 1.3)
src/spiking/populations.py        (Task #14, Phase 1.4)
src/spiking/v2_context.py         (Task #15, Phase 1.5)
src/spiking/network.py            (Task #16, Phase 1.6)
tests/test_spiking_populations.py (Task #18, Phase 1.8)
tests/test_spiking_network.py     (Task #19, Phase 1.9)
tests/test_spiking_v2.py          (Task #19, Phase 1.9)
```

Plus one allowed modification:
- `src/config.py` — add `SpikingConfig` dataclass (Task #17, Phase 1.7)

And one dependency bump:
- `requirements.txt` — add `snntorch>=0.9.0` (Task #10, Phase 1.0)

**Any file touched outside this allow-list during Phase 1 is an automatic NO-GO.** The Validator must diff against `59b6c16` before issuing a GO verdict on Phase 1.10.
