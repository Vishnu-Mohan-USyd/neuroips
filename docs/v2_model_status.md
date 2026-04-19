# V2 Model Build — Status Report

**Branch**: `v2-model-build`
**Date**: 2026-04-19
**Scope**: `src/v2_model/`, `scripts/v2/`, `tests/v2/`

This document records the current state of the v2 model — a minimal mechanistic
laminar predictive-coding circuit with local Hebbian plasticity, Kok/Richter held-out
assays, and hidden-regime procedural training. It was built and exhaustively validated
according to plan v4 (saved to `~/.claude/plans/come-to-me-with-streamed-grove.md`).

The infrastructure is complete and validated. Runtime calibration remains an
open problem: the stability-first initialization satisfies the spectral-radius guard
but is too conservative to allow the network to produce interpretable Kok/Richter
signatures in the available training budget.

---

## What was built

All modules below are implemented, tested, and pass a full pytest regression.

### `src/v2_model/` — core network (25 source files, ~3500 LoC)

| Module | Purpose |
|---|---|
| `config.py`, `state.py`, `utils.py`, `bridge.py`, `freeze_manifest.py` | Skeleton: dataclasses, NamedTuple state, Dale primitives, activation extraction, per-phase freeze manifest |
| `lgn_l4.py` | Fixed LGN ON/OFF DoG + biphasic temporal + Gabor-orientation bank (N_ori=8) and L4 E/PV divisive norm. 4×4 retinotopy × 8 orientations = 128 L4 E, 16 L4 PV. Frozen by construction (zero Parameters) |
| `connectivity.py` | Sparse retinotopic like-to-like mask generator, Cossell-2015 style, Dale-constrained softplus init |
| `plasticity.py` | Urbanczik–Senn apical-basal predictive-Hebbian, Vogels-Sprekeler iSTDP, ThresholdHomeostasis (clamp ±10), ThreeFactorRule for context-memory binding |
| `energy.py` | Circuit-wide pathway-agnostic metabolic cost: L1 rate penalty + L2 total-synaptic-current shrinkage (not feedback-specific) |
| `context_memory.py` | Module C with explicit generic/task-specific weight split per v4 required fix. τ_m = 500 ms (exp-ODE leak) |
| `layers.py` | L23E/PV/SOM + HE/PV populations, `FastInhibitoryPopulation` for τ=5 ms PV (exp-form leak), sparse like-to-like recurrence, Dale-parameterised softplus weights |
| `prediction_head.py` | Linear-plus-rectified readout from (H + C bias + L2/3 apical) → predicted next L4 E rate |
| `network.py` | Top-level V2Network wiring 245,376 Parameters (all `requires_grad=False`), synchronous Euler, phase API, frozen-sensory-core SHA invariance |
| `stimuli/feature_tokens.py` | 12 textured identity tokens, orientation-energy-balanced through LGN/Gabor bank (<5%), linear-SVM 12-way discriminability = 1.00 |
| `world/procedural.py` | Synthetic procedural world with hidden regime g (4 values, 0.98 persistence), held-out regime, train/eval seed families |

### `scripts/v2/` — drivers and eval harnesses

| Script | Purpose |
|---|---|
| `train_phase2_predictive.py` | Phase-2 pure-local predictive training on procedural world; plasticity via Urbanczik-Senn + Vogels + homeostasis + energy |
| `eval_gates.py` | Gates 1-5 (rate distribution, contrast response, surround suppression, next-step prediction vs copy-last, orientation + identity localizers) |
| `run_null_expectation_control.py` | Gate 6: null expectation control pre-Phase-3 (Kok + Richter) |
| `run_c_load_bearing_check.py` | Gate 7: ablate context memory C, confirm prediction degrades |
| `train_phase3_kok_learning.py` | Phase-3 Kok assay learning: 200ms cue → 550ms delay → 500ms probe1 → 100ms blank → 500ms probe2 (370 steps/trial), cue-mapping counterbalanced across seeds |
| `train_phase3_richter_learning.py` | Phase-3 Richter assay: 6×6 bigram matrix (Richter 2019 faithful), 500ms leader → 0ms ISI → 500ms trailer, deterministic learning + probabilistic (50%) scan |
| `eval_kok.py` | Linear SVM MVPA + mean amplitude + preferred/non-preferred asymmetry |
| `eval_richter.py` | Unit-level amplitude + RSA + preference-rank suppression + supplementary pseudo-voxel forward comparison (6-model Richter 2022) |

### `tests/v2/` — 608 tests green, 1 xfail (documented)

Structural invariants, Dale compliance, determinism, frozen-sensory-core SHA
invariance, plasticity rule sign/mask/decay, orientation-energy balance,
discriminability, procedural-world regime persistence, network phase propagation,
memory budget (4.7 MB RSS on batch=8, 100-step forward), bridge API, driver
smoke tests.

The single xfail: `test_predictive_loss_slope_is_negative_over_1000_steps`.
Reason: the current stability-first initialization is too conservative for
detectable learning in 1000 steps. Documented and targeted for future work.

---

## Status of Kok and Richter pipeline runs

### Phase 2 training (seed 42, batch=4)

| Horizon | Result |
|---|---|
| 200 steps | Stable: `\|ε\|` oscillates 0.030 → 0.029 (flat) |
| 3000 steps | Diverges: `\|ε\|` drifts 0.030 → 0.874 (30× growth) via ThresholdHomeostasis target-rate mismatch |

### Phase 3 Kok (370-step trials)

Training diverges on trial 0: `dw=4.7e26`, NaN thereafter. `eval_kok` fails with
NaN at small trial counts (2 trials → NaN in L2/3 activities → LinearSVC rejection).

### Phase 3 Richter (200-step trials)

`eval_richter` runs to completion and produces structured JSON:

- Amplitude grand_mean **2.7e27** (dominated by exploded units in tail)
- RSA `between − within` = 9.2e-4 (noise-level)
- Pseudo-voxel gain-model correlations with population mean = **1.000** across local / remote / global → all units respond nearly identically, no unit-specific modulation, no Richter signature

---

## Root cause of Kok/Richter not producing interpretable signatures

Three stacked, debugger-confirmed issues:

1. **Homeostasis target-rate mismatch** (Debugger Tasks #37, #45). `target_rate_hz = 0.5` for L23E and `0.1` for HE is unreachable from the current low-activity init; θ drifts monotonically. HOMEO-OFF isolation collapses the slope **180×** (6.9e-5 → 1.2e-6), proving homeostasis is the sole runaway driver.

2. **Cross-area loop gain** (Debugger Task #37, full-Jacobian finding). Per-population spectral radius is kept below 0.95 by the `_assert_spectral_radius_le` guard, but the 256-fan-in excitatory pathways (`W_l4_l23`, `W_l23_h`) saturate the `leak + φ'·|W_eig|` stability budget even after Task #44's fan-in-scaled init.

3. **Sub-threshold operating regime** (Task #44 consequence). Tightening excitatory weights to keep the full Jacobian < 0.95 pushed rates far below the rectified-softplus knee. The network is essentially silent (`r_l23 ≈ 0.004`, `r_h ≈ 0`), so the learning signal is too weak to overcome homeostasis drift.

The trade-off is fundamental to the current parameterisation:
- **Loose init** → explodes on 370-step trajectories.
- **Tight init** → silent, no learning.
- **Middle init** → slow monotonic drift, still divergent.

---

## Outstanding work before publishable Kok/Richter findings

Four options, from smallest to largest scope:

1. **Bounded-response homeostasis**: replace the linear `Δθ = η·(a − ρ)` rule with a saturating form that stops driving θ when activity is "close enough" to target. Removes monotonic drift. Smallest fix.
2. **Sub-phase state reset in Phase-3 drivers**: treat Kok's cue/delay/probe1/blank/probe2 as 5 independent ~50-step mini-rollouts (state resets at boundaries). Removes long-horizon accumulation without changing init.
3. **Supra-threshold calibrated init**: re-tune excitatory + inhibitory init so rates at init are in the rectified-softplus linear regime (e.g., `r_l23 ≈ 0.5` at init), accepting slight long-horizon decay that learning can correct.
4. **Add biology-motivated stability mechanisms** (synaptic depression, spike adaptation) that plan v4 deferred.

Option 1 is the minimal-effort unblock recommended as the next step.

---

## How to reproduce

```bash
# Structural test suite
PYTHONPATH=. python3 -m pytest tests/v2/ -q
# Expected: 608 passed, 1 xfailed

# Short Phase 2 training (stable, ~1 GPU-second at batch=4, 200 steps)
PYTHONPATH=. python3 -m scripts.v2.train_phase2_predictive \
    --seed 42 --n-steps 200 --batch-size 4 \
    --out-dir checkpoints/v2 \
    --held-out-regime high-hazard

# Richter eval on the checkpoint (completes; produces structured JSON; but amplitudes degenerate)
PYTHONPATH=. python3 -m scripts.v2.eval_richter \
    --checkpoint checkpoints/v2/phase2_s42/step_200.pt \
    --seed 42 --n-trials-per-condition 10 \
    --output checkpoints/v2/phase2_s42/eval_richter.json

# Kok eval on the checkpoint (fails: state diverges in 370-step trial, NaN in SVM input)
PYTHONPATH=. python3 -m scripts.v2.eval_kok \
    --checkpoint checkpoints/v2/phase2_s42/step_200.pt \
    --seed 42 --n-trials-per-condition 2 \
    --output checkpoints/v2/phase2_s42/eval_kok.json
```

---

## Key design decisions (plan v4, all honored)

- Mouse anchor throughout (populations, connectivity, target rates).
- Fixed LGN/L4 front end (no developmental Phase 1).
- Sparse like-to-like recurrence (Cossell 2015), target sparsity 12%.
- Pure-local plasticity as the main model (Urbanczik-Senn + Vogels + homeostasis + energy).
- Synthetic procedural world with hidden regime g (Gate 7 C-load-bearing precursor).
- Held-out regime in training, disjoint train/eval seed families.
- Orientation-energy-balanced identity tokens (Richter axis doesn't leak orientation signal).
- Explicit generic vs task-specific C weight split (Phase-2 / Phase-3 separation).
- Kok linear-SVM MVPA as primary decoder; Richter unit-level + pseudo-voxel forward-model comparison.
- Validator v2.3 split of Model-valid vs Claim-valid verdicts.
- Cue-mapping counterbalance across seeds for Kok.

---

## Forensic trail

- Plan file: `~/.claude/plans/come-to-me-with-streamed-grove.md` (v4, 3 review cycles)
- Debugger evidence (all with file:line citations):
  - Task #37 logs: `logs/dbg_h10.json`, `dbg_h11.json`, `dbg_h12.json`, `dbg_h13.log`
  - Task #45 logs: `logs/dbg_task45.json`, `dbg_task45.log`
  - Diagnostic harnesses: `scripts/v2/_debug_phase2_*.py`
- Incident log: `/tmp/phase2_freeze_*/INCIDENT_LOG.md` (coder-editing-mid-investigation races)
- 25+ build + validate tasks executed through a multi-agent pipeline (Researcher, Coder, Debugger, Validator, Team Lead).
